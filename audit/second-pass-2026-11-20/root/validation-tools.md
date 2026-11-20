# Validation-tool coverage gap

Reviewed snapshot69a3b19. Confirmed P2: all three validation entrypoints can report success for an empty input tree. tools/validate_gpu.py initializes failed=False and prints runtime checks passed without requiring any .cu drivers; tools/validate_host.py returns0 for0/0lessons; tools/verify_book.py treats absent docs and diagram manifest as an empty successful scan.

Reproduction used temporary directories containing only copied scripts. GPU-tool discovery stubs returned0; no CUDA execution occurred. This demonstrates the missing coverage guard, not a claim that the earlier real A100 records were fabricated or empty. The actual saved runs contain the stated examples.

Correction: fail on missing directories/manifests and zero collected cases; compare against an explicit lesson/driver manifest and verify expected counts before printing success. Record discovery failures separately from legitimate skips.

Evidence: empty-gpu-runner.txt, empty-host-runner.txt, empty-publication-check.txt in this directory.
