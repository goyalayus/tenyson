# CuTE-first interactive course redesign

## Goal
Rebuild the page so it teaches CuTE concepts first (tensor shapes, layouts, strides, tiling,
partitioning, copy atoms, retiling, and MMA fragments), then uses GPU pipeline ideas as support.
Animations must be stable and easy to run in any modern browser.

## Tech stack
- Single HTML file with embedded CSS + JS (no build step)
- DOM and CSS keyframe animations (avoid brittle canvas logic)
- Small JS state machines for stepper animations
- Prism.js for syntax highlighting

## Sections
1. CuTE mental model: shape + stride + layout as one system
2. Read-your-kernel map: each code block mapped to CuTE primitives
3. Shape algebra playground: from `shape_MNK` to `gA/gB/gC`
4. `local_tile` and `Step<>` visualization
5. `partition_S` / `partition_D` and thread slices
6. `make_tiled_copy` and async copy groups in CuTE terms
7. `partition_fragment_*` + `retile_D` + fragment flow animation
8. Pipeline written in CuTE objects (`tAgA`, `tAsA`, `tXsA`, `tXrA`)
9. Full annotated kernel (CUTE-focused walkthrough)
10. Practice checks and debugging checklist for CuTE kernels

## Animation inventory
- Matrix tile selection animation (for `local_tile`)
- Thread-to-fragment assignment animation (for `partition_*`)
- Stage ring animation for `PIPE` dimension
- Register double-buffer flip animation for `k_block`
- Interactive shape table that recomputes tensor ranks/sizes
- Code-linked stepper animation with highlighted CuTE symbols
