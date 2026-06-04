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

            // Page: the filename line, then indented page-property lines.
            let mut page = AtlasPage::new(line.trim().to_string());
            while let Some(next) = lines.peek() {
                let t = next.trim();
                // A property line is indented (legacy) or `key:value` (4.x);
                // a bare name line (neither) ends the page header.
                if t.is_empty() || (!is_indented(next) && !t.contains(':')) {
                    break;
                }
                apply_page_prop(&mut page, lines.next().unwrap().trim());
            }
            let page_index = atlas.pages.len();
            atlas.pages.push(page);

            // Regions: a non-indented name line, then indented region properties,
            // until a blank line (next page) or end of input.
            loop {
                match lines.peek() {
                    None => break,
                    Some(l) if l.trim().is_empty() => {
                        lines.next();
                        break;
                    }
                    Some(l) if is_indented(l) => {
                        lines.next(); // stray indented line; ignore
                    }
                    Some(_) => {
                        let name = lines.next().unwrap().trim().to_string();
                        let mut region = AtlasRegion::new(name, page_index);
                        while let Some(next) = lines.peek() {
                            let t = next.trim();
                            // A property line is indented (legacy) or `key:value`
                            // (4.x); a bare name line ends this region.
                            if t.is_empty() || (!is_indented(next) && !t.contains(':')) {
                                break;
                            }
                            apply_region_prop(&mut region, lines.next().unwrap().trim());
                        }
                        if region.original_width == 0 {
                            region.original_width = region.width;
                        }
                        if region.original_height == 0 {
                            region.original_height = region.height;
                        }
                        atlas.regions.push(region);
                    }
                }
            }
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

fn split_kv(line: &str) -> Option<(&str, &str)> {
    line.split_once(':').map(|(k, v)| (k.trim(), v.trim()))
}

fn parse_u32s<const N: usize>(v: &str) -> [u32; N] {
    let mut out = [0u32; N];
    for (slot, part) in out.iter_mut().zip(v.split(',')) {
        *slot = part.trim().parse().unwrap_or(0);
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
    part.and_then(|p| p.trim().parse().ok()).unwrap_or(0)
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
}
