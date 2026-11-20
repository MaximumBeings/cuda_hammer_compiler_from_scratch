# R2-11: require nonempty, complete validation coverage

Added a checked-in `tools/book-manifest.json` with the expected 43 pages, 108 actual standalone source files, 78 host C++ lessons and three CUDA generator drivers. The host, GPU, foundations and publication validators read this contract; they never regenerate it from potentially incomplete inputs. Missing directories, absent/empty manifests, missing expected lessons and unexpected source files are validation failures before compilation or GPU discovery.

The GPU runner gets its generated executable list from the explicit driver mapping rather than assuming every unexpected `.cu` file is the third lesson. Publication verification requires its nonempty diagram manifest and the existing 159-figure contract. macOS metadata and cache directories are excluded from source enumeration.

`python3 tests/validation_coverage.py` passes all 17 negative cases, including missing/empty input trees for all four entrypoints, a missing expected host/CUDA lesson, and absent/empty diagram manifests. The real checkout also matches all expected input sets. GPU discovery is stubbed only in the temporary negative fixtures; no GPU execution is claimed by this test.

The earlier displayed source count of 109 included a `__pycache__` directory in its initial collection, although it was skipped for embed comparison. The corrected manifest counts 108 actual numbered source files. Historical logs are retained as observations of the old validator; current documentation uses the corrected count.
