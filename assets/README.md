# Chine brand assets

The Chine mark is the rotate handle from a skeletal animation editor: a
bone on its pivot, with the arc you drag to turn it.

## Files

| File | Size | Use it for |
| --- | --- | --- |
| `banner.png` | 2560x1280 | The header image at the top of the README. |
| `social-preview.png` | 1280x640 | The GitHub social preview (see below). Also works for link cards and slides. |
| `icon.svg` | 512x512 | The app icon: the mark on a dark rounded tile. Use it for avatars, package registries, and documentation sites. |
| `icon-512.png` | 512x512 | The same icon as a PNG, for places that do not accept SVG. |
| `favicon-32.png` | 32x32 | A browser tab icon. |
| `mark-dark.svg` | 512x512 | The bare mark for dark backgrounds. The background is transparent. |
| `mark-light.svg` | 512x512 | The bare mark for light backgrounds. The background is transparent. |
| `lockup-dark.png` | 1384x352 | The mark with the `chine` wordmark and tagline on a dark tile. |
| `lockup-light.png` | 1384x352 | The mark with the `chine` wordmark and tagline on a light tile. |

To show the right bare mark for the reader's GitHub theme:

```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/mark-dark.svg">
  <img src="assets/mark-light.svg" alt="Chine" width="96">
</picture>
```

## Colors

| Name | Hex | Role |
| --- | --- | --- |
| Night | `#12160F` | Dark ground. |
| Bone | `#ECE6D4` | The bone, the pivot, and the wordmark on dark backgrounds. |
| Moss | `#7C9A55` | The arc on dark backgrounds, and the accent color. |
| Ink | `#1C2416` | The bone, the pivot, and the wordmark on light backgrounds. |
| Moss (light) | `#5A7A3A` | The arc on light backgrounds. |
| Sage | `#EEF2E4` | Light ground. |

The wordmark is set in Source Sans Pro, weight 600, with slightly open
letter spacing. The lockups carry the tagline "The backbone, in pure Rust."
in Source Sans Pro, weight 400.

## Social preview

GitHub does not read the social preview from the repository. To set it,
open the repository's Settings, and under General > Social preview, upload
`social-preview.png`.

## Source

The artwork is drawn in Penpot, in the Brands project, file "Repository
Brands". The mark is the "chine mark" component with Concept "CH4 Rotate
Handle" and Ink "Colour". The banner is the chine README banner board on the
Family page. Export from there if you need another size or format.
