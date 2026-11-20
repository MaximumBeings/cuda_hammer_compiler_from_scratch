# Second-pass foundations fixes (F2-01–F2-05)

All five confirmed findings were repaired in actual code, rather than resolved only by narrowing prose. Work covers the standalone sources in Parts 0–3, their embedded Markdown, focused regressions, and the foundations runner. No GPU connection or device test was performed.

| Finding | Code resolution | Verification |
| --- | --- | --- |
| F2-01 / R2-01 | Both introductory scalar interpreters now store temporary values in an id-keyed map, using checked lookup. Folding may preserve sparse SSA ids without indexing an undersized dense vector. | The complete, formerly failing 004 driver now passes AddressSanitizer. It is also a permanent focused regression, exercising the existing samples without substituting a reduced toy test. |
| F2-02 / R2-04 | Both capped fusion prepasses now count the complete candidate body, summing all inlineable predecessor subtrees plus the final operation, rather than using longest-path depth. A predecessor is materialized when it will not fit the remaining budget. Reduction Sum consumes a step too. Nonpositive budgets are rejected. The public historical argument name maxChainLength is retained, but means body-step budget. | A balanced eight-input/seven-Add tree obeys budgets 1,2,3,4,8 and retains the original results; cap zero is rejected. Existing chain/shared/reduction examples in 037/038 also pass. Chapter 16 algorithm, diagram text, questions and answers describe the new policy. |
| F2-03 / R2-05 | Nonnegative logical counts use checkedCountAdd/checkedCountMul. All copied LoopNest iteration totals check products; aggregate input/output/FLOP counts check additions; integer byte totals multiply by four with an overflow check instead of converting through double. | LLONG_MAX times two and aggregate count overflow are rejected. A valid per-tensor shape whose logical byte total exceeds long long is rejected before arithmetic or conversion becomes undefined. All original accounting examples retain their successful results. |
| F2-04 / R2-06 | Both scalar-language lexer copies check decimal accumulation against the int range. Both arithmetic helpers diagnose overflowing addition/subtraction/multiplication, division by zero and INT_MIN/-1, with the same helper used in folding and runtime evaluation. Unknown operators are rejected. | Regression checks an out-of-range literal, overflowing addition and multiplication, zero division, division overflow, and valid arithmetic. Both original scalar-compiler lessons pass sanitized execution. |
| F2-05 / R2-07 | Every static-shape inference copy in Parts 1/3 validates leaf shapes through validatedShape; it rejects nonpositive dimensions and unrepresentable products. Broadcast input and output shapes use the same validation. | Explicit rejection regressions run against all eleven inference copies: 009,010,026,029,030,031,032,033,035,036,038. This avoids relying only on the canonical Chapter 6 helper. Scalar rank-zero shapes remain valid. |

## Actual runs

- **38/38 original standalone host examples pass** with Apple Clang, C++17, `-O1 -g -Wall -Wextra -fsanitize=address,undefined,float-cast-overflow -fno-sanitize-recover=all`. Source-specific actual stdout/stderr and summary are in `validation/second-pass-fixes/foundations/`; consolidated results are `foundations-sanitized-summary.txt`.
- **21/21 focused regression suites pass** with the same sanitizers. Actual output: `foundations-regressions.txt`. This includes the original 004 driver plus new branching/counter/shape/integer counterexamples and retained first-pass checks.
- Full source fences and fresh explanatory output fences were synchronized from those successful runs. Existing chapter diagrams were preserved. The final integration checker checkpoint found no Part 0–3 source/embed mismatch; other agents' active Part 4/5/7 updates were not yet synchronized at that intermediate checkpoint (`foundations-book-check.txt`). Root owns the final whole-book check.
- `git diff --check` passed at handoff.

## Reproduction and runner behavior

```sh
python3 tests/foundations_regressions.py
SANITIZE=1 NO_SYNC=1 LOG_DIR=validation/second-pass-fixes/foundations \
  python3 tools/verify_foundations.py
```

Omit NO_SYNC only when intentionally refreshing the published source/output fences. The runner now obtains its required sources from the validated host manifest, filters Parts 0–3, and rejects empty coverage; it no longer treats an empty glob as successful validation. It writes actual compiler diagnostics, including compile failures, to logs. The original source filenames remain unchanged.

Copied shape/loop/byte patterns and helper semantics were sent directly to the systems and deployment agents for their owned later chapters. Their cross-book work and final integration checks are reported separately. This report claims local macOS/Apple Clang host correctness evidence only: it does not claim Linux reruns, GPU execution, all-input proof, optimal partitioning, measured register-pressure improvement, or complete IR validation.
