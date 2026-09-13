//! Bash-style brace expansion.
//!
//! Shard lists are conventionally written as `data-{000000..000146}.tar`, so
//! expanding brace expressions is a prerequisite for everything else. This is a
//! port of the Python `braceexpand` package that the reference implementation
//! depends on.
//!
//! ```
//! use webdataset_core::braceexpand;
//!
//! assert_eq!(
//!     braceexpand("shard-{000..002}.tar").unwrap(),
//!     ["shard-000.tar", "shard-001.tar", "shard-002.tar"],
//! );
//! assert_eq!(braceexpand("{a,b}{1,2}").unwrap(), ["a1", "a2", "b1", "b2"]);
//! ```

use crate::error::{Error, Result};
use crate::prelude::*;

/// Expand a brace expression into the list of strings it denotes.
///
/// Supports alternations (`{a,b,c}`, nestable), integer ranges with optional
/// step and zero padding (`{1..10}`, `{000..100..5}`), character ranges
/// (`{a..f}`), and backslash escapes.
///
/// A brace group that is neither an alternation nor a range is left alone, so
/// `"{abc}"` expands to itself.
pub fn braceexpand(pattern: &str) -> Result<Vec<String>> {
    let chars: Vec<char> = pattern.chars().collect();
    let groups = parse_pattern(&chars)?;
    let mut out = vec![String::new()];
    for group in groups {
        let mut next = Vec::with_capacity(out.len() * group.len());
        for prefix in &out {
            for item in &group {
                next.push(format!("{prefix}{item}"));
            }
        }
        out = next;
    }
    Ok(out)
}

/// Expand each pattern in turn and concatenate the results.
pub fn braceexpand_all<'a>(patterns: impl IntoIterator<Item = &'a str>) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for p in patterns {
        out.extend(braceexpand(p)?);
    }
    Ok(out)
}

/// Split a pattern into alternating literal runs and expanded brace groups.
fn parse_pattern(pattern: &[char]) -> Result<Vec<Vec<String>>> {
    let mut items: Vec<Vec<String>> = Vec::new();
    let mut start = 0usize;
    let mut pos = 0usize;
    let mut depth = 0usize;

    while pos < pattern.len() {
        match pattern[pos] {
            '\\' => {
                pos += 2;
                continue;
            }
            '{' => {
                if depth == 0 && pos > start {
                    items.push(vec![unescape(&pattern[start..pos])]);
                    start = pos;
                }
                depth += 1;
            }
            '}' if depth > 0 => {
                depth -= 1;
                if depth == 0 {
                    let inner = &pattern[start + 1..pos];
                    // `None` means the group is neither an alternation nor a
                    // range, so the braces stay in the output as literals.
                    if let Some(expansion) = parse_expression(inner)? {
                        items.push(expansion);
                        start = pos + 1;
                    }
                }
            }
            '}' => return Err(Error::format(format!("unbalanced braces in {:?}", to_string(pattern)))),
            _ => {}
        }
        pos += 1;
    }

    if depth != 0 {
        return Err(Error::format(format!("unbalanced braces in {:?}", to_string(pattern))));
    }
    if start < pattern.len() {
        items.push(vec![unescape(&pattern[start..])]);
    }
    if items.is_empty() {
        items.push(vec![String::new()]);
    }
    Ok(items)
}

/// Interpret the inside of a brace group, or return `None` if it is a literal.
fn parse_expression(expr: &[char]) -> Result<Option<Vec<String>>> {
    let text = to_string(expr);
    if let Some(range) = parse_int_range(&text)? {
        return Ok(Some(range));
    }
    if let Some(range) = parse_char_range(&text)? {
        return Ok(Some(range));
    }
    parse_sequence(expr)
}

/// Split a comma-separated alternation at brace depth zero and expand each arm.
fn parse_sequence(seq: &[char]) -> Result<Option<Vec<String>>> {
    let mut parts: Vec<&[char]> = Vec::new();
    let mut start = 0usize;
    let mut pos = 0usize;
    let mut depth = 0i32;

    while pos < seq.len() {
        match seq[pos] {
            '\\' => {
                pos += 2;
                continue;
            }
            '{' => depth += 1,
            '}' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(&seq[start..pos]);
                start = pos + 1;
            }
            _ => {}
        }
        pos += 1;
    }
    if depth != 0 {
        return Err(Error::format(format!("unbalanced braces in {:?}", to_string(seq))));
    }
    if parts.is_empty() {
        return Ok(None);
    }
    parts.push(&seq[start..]);

    let mut out = Vec::new();
    for part in parts {
        out.extend(braceexpand(&to_string(part))?);
    }
    Ok(Some(out))
}

/// Parse `{start..end}` or `{start..end..step}` over integers.
fn parse_int_range(expr: &str) -> Result<Option<Vec<String>>> {
    let Some((left, right, step)) = split_range(expr) else {
        return Ok(None);
    };
    if !is_int(left) || !is_int(right) {
        return Ok(None);
    }
    let step = match step {
        Some(s) => {
            let s: i64 = s.trim_start_matches('-').parse().map_err(|_| Error::format(format!("bad step {s}")))?;
            if s == 0 { 1 } else { s }
        }
        None => 1,
    };

    // Zero-padded endpoints make the whole range zero-padded, as in bash.
    let padded = [left, right].iter().any(|s| *s != "0" && *s != "-0" && (s.starts_with('0') || s.starts_with("-0")));
    let width = if padded { left.len().max(right.len()) } else { 0 };

    let start: i64 = left.parse().map_err(|_| Error::format(format!("bad range start {left}")))?;
    let end: i64 = right.parse().map_err(|_| Error::format(format!("bad range end {right}")))?;

    let mut out = Vec::new();
    if start <= end {
        let mut i = start;
        while i <= end {
            out.push(pad(i, width));
            i += step;
        }
    } else {
        let mut i = start;
        while i >= end {
            out.push(pad(i, width));
            i -= step;
        }
    }
    Ok(Some(out))
}

/// Parse `{a..f}` or `{a..f..2}` over ASCII letters.
fn parse_char_range(expr: &str) -> Result<Option<Vec<String>>> {
    let Some((left, right, step)) = split_range(expr) else {
        return Ok(None);
    };
    let (lc, rc) = match (single_letter(left), single_letter(right)) {
        (Some(l), Some(r)) => (l, r),
        _ => return Ok(None),
    };
    let step = match step {
        Some(s) => {
            let s: usize = s.trim_start_matches('-').parse().map_err(|_| Error::format(format!("bad step {s}")))?;
            if s == 0 { 1 } else { s }
        }
        None => 1,
    };

    let mut out = Vec::new();
    let (mut i, end) = (lc as i32, rc as i32);
    if lc <= rc {
        while i <= end {
            out.push(char_at(i));
            i += step as i32;
        }
    } else {
        while i >= end {
            out.push(char_at(i));
            i -= step as i32;
        }
    }
    Ok(Some(out))
}

/// Split `a..b` or `a..b..c` into its components.
fn split_range(expr: &str) -> Option<(&str, &str, Option<&str>)> {
    let parts: Vec<&str> = expr.split("..").collect();
    match parts.len() {
        2 => Some((parts[0], parts[1], None)),
        3 => Some((parts[0], parts[1], Some(parts[2]))),
        _ => None,
    }
}

fn is_int(s: &str) -> bool {
    let body = s.strip_prefix('-').unwrap_or(s);
    !body.is_empty() && body.chars().all(|c| c.is_ascii_digit())
}

fn single_letter(s: &str) -> Option<char> {
    let mut it = s.chars();
    match (it.next(), it.next()) {
        (Some(c), None) if c.is_ascii_alphabetic() => Some(c),
        _ => None,
    }
}

fn char_at(code: i32) -> String {
    char::from_u32(code as u32).map(String::from).unwrap_or_default()
}

fn pad(value: i64, width: usize) -> String {
    if width == 0 {
        return value.to_string();
    }
    if value < 0 {
        format!("-{:0>width$}", -value, width = width.saturating_sub(1))
    } else {
        format!("{value:0>width$}")
    }
}

fn to_string(chars: &[char]) -> String {
    chars.iter().collect()
}

/// Drop the backslashes that protected literal braces and commas.
fn unescape(chars: &[char]) -> String {
    let mut out = String::with_capacity(chars.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '\\' && i + 1 < chars.len() {
            out.push(chars[i + 1]);
            i += 2;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expand(p: &str) -> Vec<String> {
        braceexpand(p).unwrap()
    }

    #[test]
    fn expands_alternations() {
        assert_eq!(expand("{a,b,c}"), ["a", "b", "c"]);
        assert_eq!(expand("x{a,b}y"), ["xay", "xby"]);
        assert_eq!(expand("{a,b}{1,2}"), ["a1", "a2", "b1", "b2"]);
        assert_eq!(expand("{a,{b,c}}"), ["a", "b", "c"]);
    }

    #[test]
    fn expands_padded_integer_ranges() {
        assert_eq!(expand("{1..3}"), ["1", "2", "3"]);
        assert_eq!(expand("{000..003}"), ["000", "001", "002", "003"]);
        assert_eq!(expand("{3..1}"), ["3", "2", "1"]);
        assert_eq!(expand("{1..10..3}"), ["1", "4", "7", "10"]);
        assert_eq!(expand("{-2..2}"), ["-2", "-1", "0", "1", "2"]);
    }

    #[test]
    fn expands_character_ranges() {
        assert_eq!(expand("{a..e}"), ["a", "b", "c", "d", "e"]);
        assert_eq!(expand("{a..e..2}"), ["a", "c", "e"]);
        assert_eq!(expand("{e..a}"), ["e", "d", "c", "b", "a"]);
    }

    #[test]
    fn matches_the_shard_idiom() {
        let shards = expand("http://host/data-{000000..000003}.tar");
        assert_eq!(shards.len(), 4);
        assert_eq!(shards[0], "http://host/data-000000.tar");
        assert_eq!(shards[3], "http://host/data-000003.tar");
    }

    #[test]
    fn leaves_literals_and_non_expressions_alone() {
        assert_eq!(expand("plain.tar"), ["plain.tar"]);
        assert_eq!(expand("{abc}"), ["{abc}"]);
        assert_eq!(expand(""), [""]);
    }

    #[test]
    fn honours_escapes() {
        assert_eq!(expand(r"\{a,b\}"), ["{a,b}"]);
    }

    #[test]
    fn rejects_unbalanced_braces() {
        assert!(braceexpand("{a,b").is_err());
        assert!(braceexpand("a,b}").is_err());
    }
}
