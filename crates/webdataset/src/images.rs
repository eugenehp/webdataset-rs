//! Image decoding and encoding.
//!
//! Available with the `image` feature. The `imagespec` strings are the same
//! ones the Python library uses, so pipelines port over unchanged:
//!
//! | spec | result |
//! |---|---|
//! | `pil`, `pilrgb`, `pilrgba`, `pill` | a decoded [`image::DynamicImage`] |
//! | `l8`, `rgb8`, `rgba8` | a `u8` tensor, height × width × channels |
//! | `l`, `rgb`, `rgba` | an `f32` tensor in `0.0..=1.0`, height × width × channels |
//! | `torchl8`, `torchrgb8`, `torchrgba8` | a `u8` tensor, channels × height × width |
//! | `torchl`, `torchrgb`, `torch`, `torchrgba` | an `f32` tensor, channels × height × width |
//!
//! The `torch*` specs exist because deep learning frameworks want channels
//! first; there is no dependency on any framework.
//!
//! ```
//! # #[cfg(feature = "image")] {
//! use webdataset::images::{ImageHandler, ImageSpec};
//!
//! let spec: ImageSpec = "rgb8".parse().unwrap();
//! assert!(spec.is_tensor());
//! let handler = ImageHandler::new(spec);
//! # let _ = handler;
//! # }
//! ```

use std::str::FromStr;

use image::{DynamicImage, ImageFormat};
use webdataset_core::error::{Error, Result};
use webdataset_core::tensor::Tensor;
use webdataset_core::value::Value;

use crate::decode::{DecodeHandler, Decoded};

/// The extensions [`ImageHandler`] claims by default.
///
/// This is the list of formats the `image` crate can read, matching the set the
/// Python implementation passes to PIL.
pub const IMAGE_EXTENSIONS: &[&str] = &[
    "avif", "bmp", "dds", "exr", "ff", "gif", "hdr", "ico", "jfif", "jpe", "jpeg", "jpg", "pbm", "pgm", "pnm", "png",
    "ppm", "qoi", "tga", "tif", "tiff", "webp",
];

/// The colour space an image is converted to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMode {
    /// Single channel luminance.
    Luma,
    /// Three channels, red green blue.
    Rgb,
    /// Four channels, red green blue alpha.
    Rgba,
}

impl ColorMode {
    /// How many channels this mode has.
    pub fn channels(self) -> usize {
        match self {
            ColorMode::Luma => 1,
            ColorMode::Rgb => 3,
            ColorMode::Rgba => 4,
        }
    }
}

/// How a decoded image should be represented.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Representation {
    /// Keep it as an [`image::DynamicImage`].
    Image,
    /// A `u8` tensor, height × width × channels.
    U8Hwc,
    /// An `f32` tensor in `0.0..=1.0`, height × width × channels.
    F32Hwc,
    /// A `u8` tensor, channels × height × width.
    U8Chw,
    /// An `f32` tensor in `0.0..=1.0`, channels × height × width.
    F32Chw,
}

/// A parsed `imagespec`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageSpec {
    /// The colour space to convert to.
    pub mode: ColorMode,
    /// The representation to produce.
    pub representation: Representation,
}

impl ImageSpec {
    /// Whether this spec produces a tensor rather than an image.
    pub fn is_tensor(self) -> bool {
        self.representation != Representation::Image
    }
}

impl FromStr for ImageSpec {
    type Err = Error;

    fn from_str(spec: &str) -> Result<ImageSpec> {
        use ColorMode::{Luma, Rgb, Rgba};
        use Representation::{F32Chw, F32Hwc, Image, U8Chw, U8Hwc};
        let (mode, representation) = match spec.to_ascii_lowercase().as_str() {
            "l8" => (Luma, U8Hwc),
            "rgb8" => (Rgb, U8Hwc),
            "rgba8" => (Rgba, U8Hwc),
            "l" => (Luma, F32Hwc),
            "rgb" => (Rgb, F32Hwc),
            "rgba" => (Rgba, F32Hwc),
            "torchl8" => (Luma, U8Chw),
            "torchrgb8" => (Rgb, U8Chw),
            "torchrgba8" => (Rgba, U8Chw),
            "torchl" => (Luma, F32Chw),
            "torchrgb" | "torch" => (Rgb, F32Chw),
            "torchrgba" => (Rgba, F32Chw),
            "pill" => (Luma, Image),
            "pil" | "pilrgb" => (Rgb, Image),
            "pilrgba" => (Rgba, Image),
            other => {
                return Err(Error::value(format!(
                    "unknown imagespec {other:?}; expected one of l8 rgb8 rgba8 l rgb rgba \
                     torchl8 torchrgb8 torchrgba8 torchl torchrgb torch torchrgba pil pill pilrgb pilrgba"
                )));
            }
        };
        Ok(ImageSpec { mode, representation })
    }
}

/// Decodes image fields according to an [`ImageSpec`].
#[derive(Debug, Clone)]
pub struct ImageHandler {
    spec: ImageSpec,
    extensions: Vec<String>,
}

impl ImageHandler {
    /// Decode the standard image extensions with `spec`.
    pub fn new(spec: ImageSpec) -> ImageHandler {
        ImageHandler { spec, extensions: IMAGE_EXTENSIONS.iter().map(|e| e.to_string()).collect() }
    }

    /// Parse `spec` and build a handler.
    pub fn parse(spec: &str) -> Result<ImageHandler> {
        Ok(ImageHandler::new(spec.parse()?))
    }

    /// Claim only these extensions.
    pub fn with_extensions(mut self, extensions: impl IntoIterator<Item = impl Into<String>>) -> ImageHandler {
        self.extensions = extensions.into_iter().map(Into::into).collect();
        self
    }
}

impl DecodeHandler for ImageHandler {
    fn decode(&self, key: &str, data: &[u8]) -> Result<Decoded> {
        let extension = key.rsplit('.').next().unwrap_or(key).to_ascii_lowercase();
        if !self.extensions.contains(&extension) {
            return Ok(Decoded::Skipped);
        }
        let image = load(key, &extension, data)?;
        Ok(Decoded::Value(to_value(image, self.spec)))
    }
}

/// Decode an image, preferring libjpeg-turbo for JPEG when it is available.
fn load(key: &str, extension: &str, data: &[u8]) -> Result<DynamicImage> {
    #[cfg(feature = "libjpeg")]
    if matches!(extension, "jpg" | "jpeg" | "jpe" | "jfif") {
        return load_jpeg(key, data);
    }
    let _ = extension;
    image::load_from_memory(data).map_err(|e| Error::decode(key, e))
}

/// Decode a JPEG with libjpeg-turbo.
///
/// Decoding to RGB and converting from there, rather than asking libjpeg for
/// greyscale directly, is deliberate: that is the path Pillow takes, and the
/// two give different answers for a colour image.
#[cfg(feature = "libjpeg")]
fn load_jpeg(key: &str, data: &[u8]) -> Result<DynamicImage> {
    let decoded: turbojpeg::Image<Vec<u8>> = turbojpeg::decompress(data, turbojpeg::PixelFormat::RGB)
        .map_err(|e| Error::decode(key, format!("libjpeg: {e}")))?;
    let buffer = image::RgbImage::from_raw(decoded.width as u32, decoded.height as u32, decoded.pixels)
        .ok_or_else(|| Error::decode(key, "libjpeg returned a buffer of the wrong size"))?;
    Ok(DynamicImage::ImageRgb8(buffer))
}

/// Convert a decoded image into the representation `spec` asks for.
pub fn to_value(image: DynamicImage, spec: ImageSpec) -> Value {
    let converted = match spec.mode {
        ColorMode::Luma => DynamicImage::ImageLuma8(to_luma_601(&image)),
        ColorMode::Rgb => DynamicImage::ImageRgb8(image.to_rgb8()),
        ColorMode::Rgba => DynamicImage::ImageRgba8(image.to_rgba8()),
    };
    if spec.representation == Representation::Image {
        return Value::Image(std::sync::Arc::new(converted));
    }

    let (width, height) = (converted.width() as usize, converted.height() as usize);
    let channels = spec.mode.channels();
    let hwc = converted.into_bytes();

    // A single-channel image has no channel axis at all, which is what NumPy
    // and PIL produce and therefore what the reference implementation returns.
    let (data, shape) = match (spec.representation, channels) {
        (_, 1) => (hwc, vec![height, width]),
        (Representation::U8Hwc | Representation::F32Hwc, _) => (hwc, vec![height, width, channels]),
        _ => (to_chw(&hwc, height, width, channels), vec![channels, height, width]),
    };

    match spec.representation {
        Representation::U8Hwc | Representation::U8Chw => Value::Tensor(Tensor::from_u8_shaped(data, shape)),
        _ => {
            let scaled: Vec<f32> = data.into_iter().map(|b| b as f32 / 255.0).collect();
            Value::Tensor(Tensor::from_f32_shaped(&scaled, shape))
        }
    }
}

/// Convert to greyscale the way PIL does.
///
/// PIL applies the ITU-R 601-2 luma transform in fixed point, and the `image`
/// crate applies the Rec. 709 one, so converting with `to_luma8` would give
/// pixel values a few units away from what the reference implementation
/// produces. Matching the reference matters more here than matching the
/// ecosystem default, so the 601 coefficients are applied directly.
fn to_luma_601(image: &DynamicImage) -> image::GrayImage {
    // As in PIL's `rgb2l`: (R * 19595 + G * 38470 + B * 7471 + 0x8000) >> 16.
    // The weights sum to exactly 65536, so an already-grey pixel is unchanged.
    let rgb = image.to_rgb8();
    let (width, height) = (rgb.width(), rgb.height());
    let mut out = Vec::with_capacity((width * height) as usize);
    for pixel in rgb.pixels() {
        let [r, g, b] = pixel.0;
        let luma = (r as u32 * 19595 + g as u32 * 38470 + b as u32 * 7471 + 0x8000) >> 16;
        out.push(luma as u8);
    }
    image::GrayImage::from_raw(width, height, out).expect("the buffer was sized from the dimensions")
}

/// Reorder interleaved height × width × channels data into planar layout.
fn to_chw(hwc: &[u8], height: usize, width: usize, channels: usize) -> Vec<u8> {
    let mut out = vec![0u8; hwc.len()];
    for c in 0..channels {
        let plane = c * height * width;
        for y in 0..height {
            let row = y * width;
            for x in 0..width {
                out[plane + row + x] = hwc[(row + x) * channels + c];
            }
        }
    }
    out
}

/// Encode an image value for storage under the given extension.
pub fn encode(extension: &str, value: &Value) -> Result<Vec<u8>> {
    let format = match extension.rsplit('.').next().unwrap_or(extension).to_ascii_lowercase().as_str() {
        "jpg" | "jpeg" | "img" | "image" => ImageFormat::Jpeg,
        "png" => ImageFormat::Png,
        "ppm" | "pgm" | "pbm" | "pnm" => ImageFormat::Pnm,
        "tif" | "tiff" => ImageFormat::Tiff,
        "bmp" => ImageFormat::Bmp,
        "gif" => ImageFormat::Gif,
        "webp" => ImageFormat::WebP,
        "tga" => ImageFormat::Tga,
        "qoi" => ImageFormat::Qoi,
        other => return Err(Error::encode(extension, format!("no image encoder for {other}"))),
    };
    let image = as_image(extension, value)?;
    let mut out = std::io::Cursor::new(Vec::new());
    image.write_to(&mut out, format).map_err(|e| Error::encode(extension, e))?;
    Ok(out.into_inner())
}

/// Interpret a value as an image, converting a tensor if necessary.
fn as_image(extension: &str, value: &Value) -> Result<DynamicImage> {
    match value {
        Value::Image(image) => Ok((**image).clone()),
        Value::Tensor(tensor) => tensor_to_image(extension, tensor),
        other => Err(Error::encode(extension, format!("cannot encode {} as an image", other.type_name()))),
    }
}

/// Build an image from a height × width × channels tensor.
fn tensor_to_image(extension: &str, tensor: &Tensor) -> Result<DynamicImage> {
    let shape = tensor.shape();
    let (height, width, channels) = match shape {
        [h, w] => (*h, *w, 1),
        [h, w, c] => (*h, *w, *c),
        other => {
            return Err(Error::encode(extension, format!("expected a 2- or 3-dimensional tensor, got {other:?}")));
        }
    };

    // Floating point images are stored in 0.0..=1.0, as PIL does.
    let bytes: Vec<u8> = if tensor.dtype().is_float() {
        let values = tensor.to_f64_vec();
        if let Some(bad) = values.iter().find(|v| **v < -0.001 || **v > 1.001) {
            return Err(Error::encode(extension, format!("float image values must be in 0..1, found {bad}")));
        }
        values.into_iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect()
    } else {
        tensor.to_f64_vec().into_iter().map(|v| v.clamp(0.0, 255.0) as u8).collect()
    };

    let (w, h) = (width as u32, height as u32);
    let image = match channels {
        1 => image::GrayImage::from_raw(w, h, bytes).map(DynamicImage::ImageLuma8),
        3 => image::RgbImage::from_raw(w, h, bytes).map(DynamicImage::ImageRgb8),
        4 => image::RgbaImage::from_raw(w, h, bytes).map(DynamicImage::ImageRgba8),
        other => {
            return Err(Error::encode(extension, format!("cannot encode a {other}-channel image")));
        }
    };
    image.ok_or_else(|| Error::encode(extension, "tensor shape does not match its contents"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2x3 RGB image with distinguishable pixels.
    fn png() -> Vec<u8> {
        let mut image = image::RgbImage::new(3, 2);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            *pixel = image::Rgb([(x * 40) as u8, (y * 80) as u8, 255]);
        }
        let mut out = std::io::Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(image).write_to(&mut out, ImageFormat::Png).unwrap();
        out.into_inner()
    }

    fn decode(spec: &str) -> Value {
        match ImageHandler::parse(spec).unwrap().decode("png", &png()).unwrap() {
            Decoded::Value(value) => value,
            other => panic!("expected a value, got {other:?}"),
        }
    }

    #[test]
    fn parses_every_documented_spec() {
        for spec in ["l8", "rgb8", "rgba8", "l", "rgb", "rgba", "torchrgb", "torch", "pil", "pilrgba"] {
            assert!(spec.parse::<ImageSpec>().is_ok(), "{spec}");
        }
        assert!("nonsense".parse::<ImageSpec>().is_err());
    }

    #[test]
    fn decodes_to_height_width_channels() {
        let tensor = decode("rgb8");
        let tensor = tensor.as_tensor().unwrap();
        assert_eq!(tensor.shape(), &[2, 3, 3]);
        assert_eq!(tensor.dtype(), webdataset_core::DType::U8);
    }

    #[test]
    fn decodes_to_channels_height_width_for_torch_specs() {
        let value = decode("torchrgb8");
        assert_eq!(value.as_tensor().unwrap().shape(), &[3, 2, 3]);
    }

    #[test]
    fn scales_float_specs_into_zero_to_one() {
        let value = decode("rgb");
        let tensor = value.as_tensor().unwrap();
        assert_eq!(tensor.dtype(), webdataset_core::DType::F32);
        assert!(tensor.to_f64_vec().iter().all(|v| (0.0..=1.0).contains(v)));
    }

    #[test]
    fn keeps_the_image_for_pil_specs() {
        let value = decode("pil");
        assert!(value.as_image().is_some());
    }

    #[test]
    fn converts_colour_modes() {
        assert_eq!(decode("rgba8").as_tensor().unwrap().shape(), &[2, 3, 4]);
    }

    #[test]
    fn greyscale_uses_the_rec_601_weights() {
        // A pure red pixel: 255 * 19595 + 0x8000 >> 16 == 76, which is what PIL
        // returns and what the Rec. 709 weights would not.
        let mut image = image::RgbImage::new(1, 1);
        image.put_pixel(0, 0, image::Rgb([255, 0, 0]));
        let grey = to_luma_601(&DynamicImage::ImageRgb8(image));
        assert_eq!(grey.get_pixel(0, 0).0[0], 76);
    }

    #[test]
    fn greyscale_leaves_grey_pixels_alone() {
        let mut image = image::GrayImage::new(2, 1);
        image.put_pixel(0, 0, image::Luma([17]));
        image.put_pixel(1, 0, image::Luma([200]));
        let grey = to_luma_601(&DynamicImage::ImageLuma8(image));
        assert_eq!(grey.get_pixel(0, 0).0[0], 17);
        assert_eq!(grey.get_pixel(1, 0).0[0], 200);
    }

    #[test]
    fn greyscale_has_no_channel_axis() {
        // NumPy and PIL give a 2-D array for a single-channel image, and the
        // reference implementation passes that straight through.
        assert_eq!(decode("l8").as_tensor().unwrap().shape(), &[2, 3]);
        assert_eq!(decode("l").as_tensor().unwrap().shape(), &[2, 3]);
        assert_eq!(decode("torchl8").as_tensor().unwrap().shape(), &[2, 3]);
    }

    #[test]
    fn ignores_non_image_extensions() {
        let handler = ImageHandler::parse("rgb8").unwrap();
        assert!(matches!(handler.decode("txt", b"hello").unwrap(), Decoded::Skipped));
    }

    #[test]
    fn round_trips_through_the_encoder() {
        let decoded = decode("rgb8");
        let encoded = encode("png", &decoded).unwrap();
        let again = match ImageHandler::parse("rgb8").unwrap().decode("png", &encoded).unwrap() {
            Decoded::Value(value) => value,
            other => panic!("{other:?}"),
        };
        assert_eq!(again, decoded, "png is lossless, so the pixels must match");
    }

    #[test]
    fn rejects_out_of_range_float_images() {
        let tensor = Tensor::from_f32_shaped(&[2.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![2, 3]);
        let err = encode("png", &Value::Tensor(tensor)).unwrap_err();
        assert!(err.to_string().contains("0..1"), "{err}");
    }

    #[test]
    fn reorders_planes_correctly() {
        // Two pixels, three channels: interleaved (r0,g0,b0, r1,g1,b1).
        let hwc = vec![1, 2, 3, 4, 5, 6];
        assert_eq!(to_chw(&hwc, 1, 2, 3), vec![1, 4, 2, 5, 3, 6]);
    }
}
