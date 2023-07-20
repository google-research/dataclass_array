# Changelog

<!--

Changelog follow the https://keepachangelog.com/ standard (at least the headers)

This allow to:

* auto-parsing release notes during the automated releases from github-action:
  https://github.com/marketplace/actions/pypi-github-auto-release
* Have clickable headers in the rendered markdown

To release a new version (e.g. from `1.0.0` -> `2.0.0`):

* Create a new `# [2.0.0] - YYYY-MM-DD` header and add the current
  `[Unreleased]` notes.
* At the end of the file:
  * Define the new link url:
  `[2.0.0]: https://github.com/google-research/dataclass_array/compare/v1.0.0...v2.0.0`
  * Update the `[Unreleased]` url: `v1.0.0...HEAD` -> `v2.0.0...HEAD`

-->

## [Unreleased]

*   Add `tf.nest`/`tf.data` compatibility for `DataclassArray`.

## [1.4.2] - 2023-07-10

*   Add `dca.concat` method in addition to `dca.stack`.
*   Now require Python 3.9 (drop 3.8 support)

## [1.4.1] - 2023-03-20

*   Add `torch==2.0.0` support

## [1.4.0] - 2023-03-13

*   **Add `torch` support!**
*   Add `.cpu()`, `.cuda()`, `.to()` methods to move the dataclass from
    devices when using torch.
*   **Breaking**: `@dataclass(frozen=True)` is now automatically applied

## [1.3.0] - 2023-01-16

*   Added: Support for static `dca.DataclassArray` (dataclasses with only
    static fields).

## [1.2.1] - 2022-11-24

*   Fixed: Compatibility with `edc.dataclass(auto_cast=True)` (fix the `'type'
    object is not subscriptable` error)

## [1.2.0] - 2022-10-17

*   Changed: By default, dataclass_array do not cast and broadcast inputs
    anymore.
*   Changed: `dca.DataclassArray` fields can be annotated with named axis (e.g.
    `FloatArray['*shape h w 3']`). Note that consistency across fields is not
    checked yet.
*   Added: `@dca.dataclass_array` to customize the `dca.DataclassArray` params

## [1.1.0] - 2022-08-15

*   Added: Array types can be imported directly from `dataclass_array.typing`
*   Added: Syntax to specify the shape of the DataclassArray (e.g. `MyRay['h
    w']`).
*   Fixed: Correctly forward non-init fields in `.replace`, `tree_map`,
    `@dca.vectorize_method`

## [1.0.0] - 2022-08-08

*   Initial release

[Unreleased]: https://github.com/google-research/dataclass_array/compare/v1.4.2...HEAD
[1.4.2]: https://github.com/google-research/dataclass_array/compare/v1.4.1...v1.4.2
[1.4.1]: https://github.com/google-research/dataclass_array/compare/v1.4.0...v1.4.1
[1.4.0]: https://github.com/google-research/dataclass_array/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/google-research/dataclass_array/compare/v1.2.1...v1.3.0
[1.2.1]: https://github.com/google-research/dataclass_array/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/google-research/dataclass_array/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/google-research/dataclass_array/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/google-research/dataclass_array/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/google-research/dataclass_array/releases/tag/v0.1.0
