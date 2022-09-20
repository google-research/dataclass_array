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

*   Changed: By default, dataclass_array do not cast and broadcast inputs
    anymore.
*   Added: `@dca.dataclass_array` to customize the `dca.DataclassArray` params

## [1.1.0] - 2022-08-15

*   Added: Array types can be imported directly from `dataclass_array.typing`
*   Added: Syntax to specify the shape of the DataclassArray (e.g. `MyRay['h
    w']`).
*   Fixed: Correctly forward non-init fields in `.replace`, `tree_map`,
    `@dca.vectorize_method`

## [1.0.0] - 2022-08-08

*   Initial release

[Unreleased]: https://github.com/google-research/dataclass_array/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/google-research/dataclass_array/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/google-research/dataclass_array/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/google-research/dataclass_array/releases/tag/v0.1.0
