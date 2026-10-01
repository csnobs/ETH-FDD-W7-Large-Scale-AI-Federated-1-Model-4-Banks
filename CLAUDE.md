# One Model, Four Banks — ETH FDD Week 7 project

A single Jupyter notebook ([we7_project_federated_learning_student-cn.ipynb](we7_project_federated_learning_student-cn.ipynb))
for week 7 of an ETH Zurich CAS course (repo `eth-fdd-fs26/FDD-WE7-public`, "FS26" = spring
semester 2026). It teaches federated learning end to end using a scenario of four fictional
banks that want one shared credit-risk model without pooling their customer records.

This is a course exercise, not a library — there is no application code to build or deploy,
only the notebook and the support modules it pulls in at runtime.

## How the notebook is structured

Designed to run top to bottom in Google Colab (badge in the first cell), though it also runs
locally with Jupyter + numpy. It walks through seven parts:

1. **Train alone, or train together?** — compare each bank's local model against a pooled
   reference on AUC.
2. **Do the banks see the same customers?** — check whether the four banks are learning the
   same underlying rule.
3. **Training together without sharing data** — derive FedSGD/FedAvg from first principles:
   averaging per-bank gradients weighted by book size reproduces the pooled gradient exactly.
4. **Train more locally, communicate less** — the cost of communication rounds vs. local steps
   per exchange (FedAvg's `E` parameter).
5. **Can an update reveal a customer?** — gradient inversion: reconstructing an exact customer
   record from a single-record update, and how that attack degrades as batch size grows.
6. **How do we protect customers?** — differential privacy via per-record gradient clipping
   (a cap `C`) plus calibrated Gaussian noise (`z`), and the epsilon/utility tradeoff. This is
   where the **required** part of the notebook ends.
7. *(optional)* **What happens at real scale?** — model size, federation size, and payload
   compression (top-k sparsification, sign compression) for larger models/federations than the
   4-bank toy case.

Six cells contain `???` blanks marked with a `🎯 Task` comment (Tasks 1–6); five are required
(parts 1–6), one is optional (part 7). Each graded cell calls `fv.check(...)` to validate the
answer inline and point at what's wrong. Four more cells are hand-filled tables (**✍️ Fill this
in**) for qualitative reflection. Parts 3, 5, and 6 each end in an interactive "mission"
playground (`fv.mission(...)`) rather than a task cell.

## Runtime dependencies (fetched, not vendored)

The first code cell (0.1) clones or downloads five support modules from
`eth-fdd-fs26/FDD-WE7-public` (`project/src/`) into a local `src/` directory at run time and adds
it to `sys.path`:

- `fedcore.py` (imported as `fc`) — the federated-learning core: model init/gradient/loss/AUC,
  `fc.run(...)` (FedAvg loop), `fc.aggregate_weighted`, `fc.per_record_gradients`,
  `fc.Privacy`/`fc.epsilon_per_site` (DP), `fc.Federation`/`fc.Site`.
- `fedviz.py` (imported as `fv`) — all plotting/visualization and the `fv.check(...)` grader /
  `fv.mission(...)` playgrounds.
- `generator.py` (imported as `gen`) — synthetic data generation (`gen.generate`, `gen.TRUE_BETA`,
  `gen.REFERENCE`, `gen.BANKS`, `gen.BOOK_SEED`).
- `scenario.py` — the fixed four-bank scenario built from the generator (`scenario.split()`,
  `scenario.load()`, `scenario.load_raw()`, feature names, business labels).
- `quizzes.py` — not directly referenced in the cells inspected, likely used inside `fedviz`'s
  missions.

None of these files are checked into this repo; `src/` is git-ignored-by-absence and recreated
by running the notebook's first cell. **The customer data is entirely synthetic** (14 standardized
features, generated from a known ground-truth rule `gen.TRUE_BETA`), which is what makes exact
gradient-inversion reconstruction in part 5 possible to verify.

## Working with this notebook

- There's no build/test/lint tooling here — correctness is checked by running the notebook and
  reading each task cell's `fv.check(...)` output.
- If editing or reviewing task-cell logic, the reference implementations the checks are built
  against are visible from the `fv.check` expected/hint strings in each cell — treat those as
  the spec.
- Model is a plain logistic regression over 13 standardized features + bias (`fc.N_FEATURES`,
  `fc.N_PARAMS`); nothing deep-learning-specific, by design — the point is the FL/privacy
  mechanics, not the model architecture.

## Repo history note

This repo was initialized here (scoped to this project folder) after being accidentally
`git init`'d at the user's home directory (`~/.git`) in an earlier session. That stray home-dir
repo still exists and tracking it as a separate (unrelated) cleanup item — see git log for
context if it resurfaces.
