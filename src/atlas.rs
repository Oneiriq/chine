//! Spine texture-atlas (`.atlas`) parsing.
//!
//! An [`Atlas`] describes how a skeleton's images are packed into one or more
//! texture [`AtlasPage`]s. `chine` parses the metadata; loading the page images
//! (by [`AtlasPage::name`]) and creating GPU textures is left to the host.
//!
//! The current (Spine 4.1+) text format is supported, plus the common legacy
//! keys (`xy`/`size`/`orig`/`offset`) as a fallback.

/// Texture minification / magnification filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AtlasFilter {
    /// Nearest-neighbor sampling.
    #[default]
    Nearest,
    /// Bilinear sampling.
    Linear,
    /// Trilinear mipmap sampling.
    MipMap,
    /// Mipmap, nearest within and between levels.
    MipMapNearestNearest,
    /// Mipmap, linear within levels, nearest between.
    MipMapLinearNearest,
    /// Mipmap, nearest within levels, linear between.
    MipMapNearestLinear,
    /// Mipmap, linear within and between levels.
    MipMapLinearLinear,
}

/// Texture wrap mode along one axis (from the atlas `repeat` key).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AtlasWrap {
    /// Clamp to the edge (no repeat).
    #[default]
    ClampToEdge,
    /// Repeat (tile).
    Repeat,
}

/// One texture page: an image file plus the page-wide sampler settings.
#[derive(Debug, Clone)]
pub struct AtlasPage {
    /// Image filename, relative to the atlas file (the host loads this).
    pub name: String,
    /// Page width in pixels.
    pub width: u32,
    /// Page height in pixels.
    pub height: u32,
    /// Whether the page texture uses premultiplied alpha.
    pub pma: bool,
    /// Minification filter.
    pub min_filter: AtlasFilter,
    /// Magnification filter.
    pub mag_filter: AtlasFilter,
    /// Horizontal wrap mode.
    pub wrap_u: AtlasWrap,
    /// Vertical wrap mode.
    pub wrap_v: AtlasWrap,
}

impl AtlasPage {
    fn new(name: String) -> Self {
        Self {
            name,
            width: 0,
            height: 0,
            pma: false,
            min_filter: AtlasFilter::Nearest,
            mag_filter: AtlasFilter::Nearest,
            wrap_u: AtlasWrap::ClampToEdge,
            wrap_v: AtlasWrap::ClampToEdge,
        }
    }
}

/// One packed image region within a page. An attachment's `path` names a
/// region; its UVs come from the packed rect, its layout from the offsets.
#[derive(Debug, Clone)]
pub struct AtlasRegion {
    /// Region name (the attachment `path` references this).
    pub name: String,
    /// Index of the owning page in [`Atlas::pages`].
    pub page: usize,
    /// Packed x in the page, in pixels.
    pub x: u32,
    /// Packed y in the page, in pixels.
    pub y: u32,
    /// Packed width in pixels (the rotated extent when `degrees` is 90 / 270).
    pub width: u32,
    /// Packed height in pixels.
    pub height: u32,
    /// Counter-clockwise rotation of the region within the page: 0, 90, 180, 270.
    pub degrees: u16,
    /// Left whitespace stripped during packing, in pixels.
    pub offset_x: f32,
    /// Bottom whitespace stripped during packing, in pixels.
    pub offset_y: f32,
    /// Original (pre-pack) width, in pixels.
    pub original_width: u32,
    /// Original (pre-pack) height, in pixels.
    pub original_height: u32,
    /// Animation frame index, or `-1` if unindexed.
    pub index: i32,
}

impl AtlasRegion {
    fn new(name: String, page: usize) -> Self {
        Self {
            name,
            page,
            x: 0,
            y: 0,
            width: 0,
            height: 0,
            degrees: 0,
            offset_x: 0.0,
            offset_y: 0.0,
            original_width: 0,
            original_height: 0,
            index: -1,
        }
    }
}

/// A parsed Spine texture atlas: its pages and the regions packed across them.
#[derive(Debug, Clone, Default)]
pub struct Atlas {
    /// Texture pages, in file order.
    pub pages: Vec<AtlasPage>,
    /// Regions across all pages, in file order.
    pub regions: Vec<AtlasRegion>,
}

impl Atlas {
    /// Parse an atlas from its `.atlas` text contents.
    #[must_use]
    pub fn parse(text: &str) -> Self {
        let mut atlas = Self::default();
        let mut lines = text.lines().peekable();

        while let Some(line) = lines.next() {
            if line.trim().is_empty() {
                continue; // blank lines separate pages
            }

            // Page: the filename line, then its property lines.
            let mut page = AtlasPage::new(line.trim().to_string());
            while let Some(prop) = lines.next_if(|l| is_property(l)) {
                apply_page_prop(&mut page, prop.trim());
            }
            let page_index = atlas.pages.len();
            atlas.pages.push(page);

            // Regions: a name line, then its property lines, until a blank line
            // (next page) or end of input. The property loops consume every
            // indented or `key:value` line, so each line reached here is a name.
            while let Some(name) = lines.next_if(|l| !l.trim().is_empty()) {
                let mut region = AtlasRegion::new(name.trim().to_string(), page_index);
                while let Some(prop) = lines.next_if(|l| is_property(l)) {
                    apply_region_prop(&mut region, prop.trim());
                }
                if region.original_width == 0 {
                    region.original_width = region.width;
                }
                if region.original_height == 0 {
                    region.original_height = region.height;
                }
                atlas.regions.push(region);
            }
            // Consume the blank line that ends the page, if any.
            lines.next();
        }

        atlas
    }

    /// Find a region by name (the first one declared with that name).
    #[must_use]
    pub fn find_region(&self, name: &str) -> Option<&AtlasRegion> {
        self.regions.iter().find(|r| r.name == name)
    }
}

fn is_indented(line: &str) -> bool {
    line.starts_with([' ', '\t'])
}

/// A property line is indented (legacy) or `key:value` (4.x). A blank line or
/// a bare name line (neither) is not.
fn is_property(line: &str) -> bool {
    let t = line.trim();
    !t.is_empty() && (is_indented(line) || t.contains(':'))
}

fn split_kv(line: &str) -> Option<(&str, &str)> {
    line.split_once(':').map(|(k, v)| (k.trim(), v.trim()))
}

/// Parse a pixel size or position. Spine reads these as signed 32-bit
/// integers, so a value past `i32::MAX` is invalid and reads as 0, like any
/// other unparsable value. The limit also keeps `x + width` and
/// `y + height` within `u32`.
fn parse_pixels(part: &str) -> u32 {
    part.trim()
        .parse::<u32>()
        .ok()
        .filter(|&n| i32::try_from(n).is_ok())
        .unwrap_or(0)
}

fn parse_u32s<const N: usize>(v: &str) -> [u32; N] {
    let mut out = [0u32; N];
    for (slot, part) in out.iter_mut().zip(v.split(',')) {
        *slot = parse_pixels(part);
    }
    out
}

fn parse_filter(s: &str) -> AtlasFilter {
    match s.trim() {
        "Linear" => AtlasFilter::Linear,
        "MipMap" => AtlasFilter::MipMap,
        "MipMapNearestNearest" => AtlasFilter::MipMapNearestNearest,
        "MipMapLinearNearest" => AtlasFilter::MipMapLinearNearest,
        "MipMapNearestLinear" => AtlasFilter::MipMapNearestLinear,
        "MipMapLinearLinear" => AtlasFilter::MipMapLinearLinear,
        _ => AtlasFilter::Nearest,
    }
}

fn parse_rotate(v: &str) -> u16 {
    match v {
        "true" => 90,
        "false" => 0,
        other => other.parse().unwrap_or(0),
    }
}

fn apply_page_prop(page: &mut AtlasPage, line: &str) {
    let Some((key, value)) = split_kv(line) else {
        return;
    };
    match key {
        "size" => {
            let [w, h] = parse_u32s::<2>(value);
            page.width = w;
            page.height = h;
        }
        "filter" => {
            let mut parts = value.split(',');
            page.min_filter = parts.next().map(parse_filter).unwrap_or_default();
            page.mag_filter = parts.next().map(parse_filter).unwrap_or_default();
        }
        "pma" => page.pma = value == "true",
        "repeat" => {
            page.wrap_u = if value == "x" || value == "xy" {
                AtlasWrap::Repeat
            } else {
                AtlasWrap::ClampToEdge
            };
            page.wrap_v = if value == "y" || value == "xy" {
                AtlasWrap::Repeat
            } else {
                AtlasWrap::ClampToEdge
            };
        }
        _ => {} // `format` and unknowns are ignored
    }
}

fn apply_region_prop(region: &mut AtlasRegion, line: &str) {
    let Some((key, value)) = split_kv(line) else {
        return;
    };
    match key {
        "index" => region.index = value.parse().unwrap_or(-1),
        "rotate" => region.degrees = parse_rotate(value),
        // 4.1+ packed rect.
        "bounds" => {
            let [x, y, w, h] = parse_u32s::<4>(value);
            region.x = x;
            region.y = y;
            region.width = w;
            region.height = h;
        }
        // 4.1+ offsets: left, bottom, originalWidth, originalHeight.
        "offsets" => {
            let mut parts = value.split(',');
            region.offset_x = parse_f32(parts.next());
            region.offset_y = parse_f32(parts.next());
            region.original_width = parse_u32(parts.next());
            region.original_height = parse_u32(parts.next());
        }
        // Legacy keys.
        "xy" => {
            let [x, y] = parse_u32s::<2>(value);
            region.x = x;
            region.y = y;
        }
        "size" => {
            let [w, h] = parse_u32s::<2>(value);
            region.width = w;
            region.height = h;
        }
        "orig" => {
            let [w, h] = parse_u32s::<2>(value);
            region.original_width = w;
            region.original_height = h;
        }
        "offset" => {
            let mut parts = value.split(',');
            region.offset_x = parse_f32(parts.next());
            region.offset_y = parse_f32(parts.next());
        }
        _ => {}
    }
}

fn parse_f32(part: Option<&str>) -> f32 {
    part.and_then(|p| p.trim().parse().ok()).unwrap_or(0.0)
}

fn parse_u32(part: Option<&str>) -> u32 {
    part.map_or(0, parse_pixels)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "page1.png\n\tsize: 640,480\n\tformat: RGBA8888\n\tfilter: Linear,Linear\n\trepeat: none\n\tpma: true\ndagger\n\tbounds: 372,100,26,108\nhead\n\tindex: 0\n\tbounds: 2,21,103,81\n\trotate: 90\n\noffhand.png\n\tsize: 64,64\n\trepeat: x\nshield\n\tbounds: 0,0,40,40\n\toffsets: 2,2,44,44\n";

    #[test]
    fn parses_pages_and_regions() {
        let atlas = Atlas::parse(SAMPLE);
        assert_eq!(atlas.pages.len(), 2);
        assert_eq!(atlas.regions.len(), 3);

        let p0 = &atlas.pages[0];
        assert_eq!(p0.name, "page1.png");
        assert_eq!((p0.width, p0.height), (640, 480));
        assert!(p0.pma);
        assert_eq!(p0.min_filter, AtlasFilter::Linear);
        assert_eq!(p0.wrap_u, AtlasWrap::ClampToEdge);

        // page2 repeats on x.
        assert_eq!(atlas.pages[1].name, "offhand.png");
        assert_eq!(atlas.pages[1].wrap_u, AtlasWrap::Repeat);
        assert_eq!(atlas.pages[1].wrap_v, AtlasWrap::ClampToEdge);
    }

    #[test]
    fn region_fields_and_page_linkage() {
        let atlas = Atlas::parse(SAMPLE);
        let head = atlas.find_region("head").unwrap();
        assert_eq!(head.page, 0);
        assert_eq!((head.x, head.y, head.width, head.height), (2, 21, 103, 81));
        assert_eq!(head.degrees, 90);
        assert_eq!(head.index, 0);
        // original size defaults to packed size when no offsets given.
        assert_eq!(head.original_width, 103);

        // `dagger` had no index/rotate -> defaults.
        let dagger = atlas.find_region("dagger").unwrap();
        assert_eq!(dagger.index, -1);
        assert_eq!(dagger.degrees, 0);

        // `shield` is on page 2 with explicit offsets/original size.
        let shield = atlas.find_region("shield").unwrap();
        assert_eq!(shield.page, 1);
        assert_eq!((shield.original_width, shield.original_height), (44, 44));
        assert!((shield.offset_x - 2.0).abs() < 1e-6);
    }

    #[test]
    fn missing_region_is_none() {
        assert!(Atlas::parse(SAMPLE).find_region("nope").is_none());
    }

    /// Every region's packed rect and original size, as `(x + w, y + h)` sums
    /// that must not overflow.
    fn rect_ends(atlas: &Atlas) -> Vec<Option<(u32, u32)>> {
        atlas
            .regions
            .iter()
            .map(|r| Some((r.x.checked_add(r.width)?, r.y.checked_add(r.height)?)))
            .collect()
    }

    #[test]
    fn pixel_values_past_i32_max_read_as_zero() {
        // A fuzzed atlas whose rect sums overflowed `u32` when binding UVs.
        let text = "p.png\nsize: 4294967295,1\nr\n  bounds: 4294967295,4294967295,4294967295,4294967295\n  offsets: -1e30,1e30,4294967295,0\n  rotate: 65535\nm\n  bounds: 1,2\n  rotate: -90\ns0\n  index: 99999999999\n";
        let atlas = Atlas::parse(text);
        assert_eq!(atlas.pages.len(), 1);
        assert_eq!(atlas.pages[0].width, 0);
        assert_eq!(atlas.regions.len(), 3);
        let r = atlas.find_region("r").unwrap();
        assert_eq!((r.x, r.y, r.width, r.height), (0, 0, 0, 0));
        assert_eq!((r.original_width, r.original_height), (0, 0));
        assert!(rect_ends(&atlas).iter().all(Option::is_some));
        // Unparsable index and rotate values keep their defaults.
        assert_eq!(atlas.find_region("s0").unwrap().index, -1);
        assert_eq!(atlas.find_region("m").unwrap().degrees, 0);
    }

    #[test]
    fn largest_i32_pixel_values_are_kept_and_sum_in_range() {
        let atlas =
            Atlas::parse("p.png\nr\n  bounds: 2147483647,2147483647,2147483647,2147483647\n");
        let r = &atlas.regions[0];
        assert_eq!((r.x, r.width), (2_147_483_647, 2_147_483_647));
        assert!(rect_ends(&atlas).iter().all(Option::is_some));
    }

    #[test]
    fn odd_line_layouts_end_pages_and_regions_cleanly() {
        // CRLF endings, an indented line without a colon, a page with no
        // regions, and a property on the last line with no final newline.
        let text = "a.png\r\n  size: 8,8\r\n  stray\r\n\r\nb.png\nsize: 4,4\nr\n  bounds: 1,1,2,2\n  rotate: true\n\nc.png\n  size: 2,2";
        let atlas = Atlas::parse(text);
        let names: Vec<&str> = atlas.pages.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["a.png", "b.png", "c.png"]);
        assert_eq!((atlas.pages[0].width, atlas.pages[2].height), (8, 2));
        assert_eq!(atlas.regions.len(), 1);
        assert_eq!(atlas.regions[0].page, 1);
        assert_eq!(atlas.regions[0].degrees, 90);
    }
}
