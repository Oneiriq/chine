# Changelog

## 0.2.0 (unreleased)

### Added

- `BoneData::skin_required`. Struct literals without `..Default::default()`
  need it.
- `Bone::active` reports whether a bone applies with the active skin.
- `Bone::inherit` reports a bone's current inherit mode.
- `Skeleton::clear_events` is public. A host that calls `Animation::apply`
  directly calls it first.
- Bone inherit timelines. A `.skel` export that used them failed to load.
- Draw order folder timelines.
- Convex and inverse clipping attachments.
- chine-web: `WebSpine::skin_names`, `WebSpine::set_skin`, and a `skin`
  attribute on `<chine-spine>` and `<spine-skeleton>`. Spine's element
  combines a comma-separated list of skins. chine-web shows the first one.

### Changed

- Skin-required bones and constraints apply only while the active skin lists
  them. `Skeleton::set_skin` rebuilds the update cache.
- Deform and sequence timelines look only in the skin they name, as Spine
  does.
- The JSON loader reads constraints before skins, so skins can name them.
- Constraint timelines leave inactive constraints alone. Global physics
  timelines and physics resets touch only active constraints.
- Linked meshes resolve against the skin, slot, and name they give. A link
  without a skin resolves in the default skin. JSON reads the Spine 4.3
  `source`, `slot`, and `timelines` keys. The older `parent` and `deform` keys
  still load.
- Deform timelines also drive path, bounding box, and clipping vertices.
- Path constraints follow the path the slot shows. A path without constant
  speed is sampled by its exported curve lengths.
- The `json` feature enables only `serde_json`. The unused `serde` dependency
  is gone.

### Fixed

- `.skel`: a bone's inherit mode is read before its length. The official 4.3
  examples loaded wrong bone lengths and inherit modes.
- `.skel`: constraint timelines index the single list of every constraint.
- `.skel`: unset IK, transform, and slider mixes read as 0.
- `.skel`: the header bounds size is kept in `SkeletonData::size`.
- `.skel`: linked meshes keep their source slot, skin, and sequence.
- JSON: path position and spacing timelines read their keyed values. A key
  without `mixY` takes its `mixX`.
- JSON: deform timelines nested under `attachments` load.
- JSON: slider constraints load.
- JSON: a sequence key without a delay keeps the previous key's delay.
- JSON: an attachment's path defaults to its `name`.
- A looping slider wraps its time into the animation. It held the last frame.
- The bones a slider animates are computed again after it.
- A weighted path's bones are computed before its constraint.
- A clip that starts while another is active is ignored, as in Spine.
