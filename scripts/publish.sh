#!/usr/bin/env bash
#
# Publish the Rust crates to crates.io, in dependency order, one at a time.
#
# `cargo publish --workspace` is all-or-nothing: when one crate is rejected the
# whole command aborts, and anything already uploaded stays uploaded. That is
# how this workspace ended up half-published once. Publishing crate by crate
# and skipping versions that are already up makes the operation resumable —
# fix the cause, run it again, and it carries on from where it stopped.
#
# Publishing cannot be undone. A version can be yanked, but the name and
# version are taken for good. So this runs a dry run unless told otherwise.
#
#   scripts/publish.sh                 # check everything, upload nothing
#   scripts/publish.sh --execute       # actually publish
#   scripts/publish.sh --skip-checks   # when the checks have already run
#
set -euo pipefail

cd "$(dirname "$0")/.."

# Dependency order. Each crate must be on the index before the next one can
# build against it, so the order is not cosmetic.
CRATES="webdataset-core webdataset-tenbin webdataset-io webdataset-shard webdataset webdataset-cli"

EXECUTE=0
SKIP_CHECKS=0
ALLOW_DIRTY=0

for arg in "$@"; do
    case "$arg" in
        --execute)      EXECUTE=1 ;;
        --skip-checks)  SKIP_CHECKS=1 ;;
        --allow-dirty)  ALLOW_DIRTY=1 ;;
        -h|--help)      sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)              echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[31merror: %s\033[0m\n' "$*" >&2; exit 1; }

VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)
[ -n "$VERSION" ] || die "could not read the version from Cargo.toml"

say "Publishing version $VERSION"
if [ "$EXECUTE" -eq 1 ]; then
    info "MODE: live — uploads are permanent"
else
    info "MODE: dry run — pass --execute to upload"
fi

# --- checks -----------------------------------------------------------------

if [ "$SKIP_CHECKS" -eq 0 ]; then
    say "Version is consistent across the workspace"
    # Workspace members inherit the version, so they cannot drift as long as
    # they keep inheriting it. What can drift is checked here: a member that
    # stopped inheriting, the path-dependency pins that Cargo turns into version
    # requirements when it packages, and the bindings crate, which sits outside
    # the workspace and carries a version of its own.
    mismatch=0

    for manifest in crates/*/Cargo.toml; do
        case "$manifest" in
            crates/webdataset-python/Cargo.toml) continue ;;
        esac
        if ! grep -q '^version.workspace = true' "$manifest"; then
            own=$(sed -n 's/^version = "\([0-9][0-9.]*\)"/\1/p' "$manifest" | head -1)
            if [ "$own" != "$VERSION" ]; then
                info "$manifest does not inherit the version and declares ${own:-nothing}"
                mismatch=1
            fi
        fi
    done

    # The `version =` half of each `{ version = ..., path = "crates/..." }` pin.
    # Cargo rewrites path dependencies into version requirements when packaging,
    # so a stale pin here is what downstream users would try to resolve.
    for found in $(grep -oE 'version = "[0-9][0-9.]*", path = "crates/' Cargo.toml | grep -oE '[0-9][0-9.]+'); do
        if [ "$found" != "$VERSION" ]; then
            info "a path-dependency pin in Cargo.toml is $found, expected $VERSION"
            mismatch=1
        fi
    done

    # The bindings crate: its own version, and its requirement on `webdataset`.
    for found in $(grep -oE '^version = "[0-9][0-9.]*"|version = "[0-9][0-9.]*", path = "\.\./webdataset"' \
                   crates/webdataset-python/Cargo.toml | grep -oE '[0-9][0-9.]+'); do
        if [ "$found" != "$VERSION" ]; then
            info "webdataset-python declares $found, expected $VERSION"
            mismatch=1
        fi
    done

    [ "$mismatch" -eq 0 ] || die "version mismatch; fix the manifests above"
    info "members inherit it; path pins and the bindings crate agree on $VERSION"

    say "Working tree is clean"
    if git rev-parse --git-dir >/dev/null 2>&1; then
        if [ -n "$(git status --porcelain)" ]; then
            if [ "$ALLOW_DIRTY" -eq 1 ]; then
                info "tree is dirty, continuing because --allow-dirty was given"
            else
                git status --short | sed 's/^/    /'
                die "uncommitted changes; commit them or pass --allow-dirty"
            fi
        else
            info "clean"
        fi
    else
        info "not a git repository, skipping"
        ALLOW_DIRTY=1
    fi

    say "Tests"
    cargo test --workspace --all-features >/dev/null
    info "passed"

    say "Lints and formatting"
    cargo clippy --workspace --all-features --all-targets -- -D warnings >/dev/null 2>&1
    cargo fmt --all --check
    info "clean"

    say "Documentation builds"
    RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps >/dev/null 2>&1
    info "clean"
else
    say "Checks skipped (--skip-checks)"
fi

# --- credentials ------------------------------------------------------------

if [ "$EXECUTE" -eq 1 ]; then
    say "Credentials"
    if [ -n "${CARGO_REGISTRY_TOKEN:-}" ]; then
        info "using CARGO_REGISTRY_TOKEN from the environment"
    elif [ -f "${CARGO_HOME:-$HOME/.cargo}/credentials.toml" ]; then
        info "using the token in ${CARGO_HOME:-$HOME/.cargo}/credentials.toml"
    else
        die "no crates.io token; run 'cargo login' or set CARGO_REGISTRY_TOKEN"
    fi
fi

# --- is this version already on crates.io? ----------------------------------

# Returns 0 when $1 already exists at $VERSION, so it can be skipped.
already_published() {
    local crate="$1" body
    body=$(curl -sS --max-time 30 \
        -H "User-Agent: webdataset-rs publish script" \
        "https://crates.io/api/v1/crates/$crate/$VERSION" 2>/dev/null) || return 1
    case "$body" in
        *'"errors"'*) return 1 ;;
        *'"num"'*)    return 0 ;;
        *)            return 1 ;;
    esac
}

# crates.io serves the index through a CDN, so a crate can be accepted a moment
# before the next crate's build can resolve it. Cargo waits for its own upload,
# but the wait is cheap insurance and makes a failure legible when it is not.
wait_for_index() {
    local crate="$1" waited=0
    while [ "$waited" -lt 120 ]; do
        already_published "$crate" && { info "$crate@$VERSION is on the index"; return 0; }
        sleep 5
        waited=$((waited + 5))
    done
    die "$crate@$VERSION did not appear on the index within 120s"
}

# --- publish ----------------------------------------------------------------

say "Publishing"

# The two modes differ on purpose.
#
# Verifying has to be done for the workspace at once. A crate cannot be checked
# on its own until everything it depends on is on the index, so verifying
# `webdataset` before `webdataset-shard` is published is impossible in
# isolation — `--workspace` resolves the unpublished crates against each other.
#
# Uploading has to be done one crate at a time, for the opposite reason:
# `--workspace` aborts the whole run on the first rejection and keeps whatever
# it already sent, which is how this workspace ended up half-published. Crate by
# crate, a rejection stops at that crate and the next run resumes.

pending=""
skipped=0
for crate in $CRATES; do
    if already_published "$crate"; then
        info "skip    $crate@$VERSION — already on crates.io"
        skipped=$((skipped + 1))
    else
        info "to do   $crate@$VERSION"
        pending="$pending $crate"
    fi
done

if [ -z "$pending" ]; then
    say "Done"
    info "every crate is already published at $VERSION"
    exit 0
fi

count=$(echo $pending | wc -w | tr -d ' ')

dirty_arg=""
[ "$ALLOW_DIRTY" -eq 1 ] && dirty_arg="--allow-dirty"

if [ "$EXECUTE" -eq 0 ]; then
    say "Verifying (whole workspace, so unpublished crates resolve against each other)"
    # shellcheck disable=SC2086
    cargo publish --dry-run --workspace $dirty_arg
    say "Done"
    info "verified $count, already up $skipped"
    info "nothing was uploaded; re-run with --execute to publish"
    exit 0
fi

say "Uploading $count crate(s), in dependency order"
published=0
for crate in $pending; do
    info "publish $crate@$VERSION"
    # shellcheck disable=SC2086
    cargo publish -p "$crate" $dirty_arg
    wait_for_index "$crate"
    published=$((published + 1))
done

say "Done"
info "published $published, already up $skipped"
info ""
info "The Python package is separate — it goes to PyPI, not crates.io:"
info "  cd crates/webdataset-python && maturin build --release --out dist"
info "  twine upload dist/*"
info ""
info "Wheels are per-platform. Build them on each target, or in CI, so"
info "the release is not just the machine that happened to run this."
