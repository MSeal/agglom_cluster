agglom_cluster Changelog
=======================
2.0.1 - 05/16/2016
### Bug
- Fixed bug where graphs with integer nodes would sometimes generate damaged dendrograms

### Improvement
- Fixed line-endings and added gitattributes for better cross-platform development

2.0.0 - 01/30/2016
------------------
### Improvements
- Added cython compilation for speedup
- Refactored model into more intuitive design
- Removed redundant code paths/repeated copies

1.0.3 - 08/24/2015
------------------
### Improvements
- Added ability to pass a uniqueness flag that forces all quality scores between clusters with the
flag to be 0.
