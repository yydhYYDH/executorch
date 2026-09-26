# BOOLACCT — the boolean/comparison island, final accounting

**Status: IN PROGRESS. Numbers below are placeholders until the measurement run lands.**

**Evidence tier for this whole file: host lowering.** Export +
`to_edge_transform_and_lower` with the Hexagon partitioner, the support predicate, and
the AOT blob read with the runtime's own `read_blob`. No kernel ran, `hexagon-sim`
did not run, and no phone was involved — so no skel md5 is quoted and no kernel claim
is made here.

**Trees, verified by `git rev-parse`:**

| branch | sha |
|---|---|
| `hexagon-mergeq` (measurement base) | `1e59f07` |
| `hexagon-integration-a47` | `1e07607` |
| `hexagon-gateinv` (unmerged) | `ee4471b` |
| this branch | `hexagon-boolacct` off `1e59f07` |

**Worktree:** `.tmp/wt/boolacct/executorch` (leaf basename `executorch`, as
`CMakeLists.txt:532` requires).

**Generated untracked `.fbs` files present in this worktree**, copied from tracked
`schema/` because a fresh worktree has no build:

    87710ba26344a6dd925e6ade1ab69e82  schema/program.fbs       == exir/_serialize/program.fbs
    dd18584af8a59f59c638a40b5955dc41  schema/scalar_type.fbs   == exir/_serialize/scalar_type.fbs

`cmp` reports both pairs byte-identical. Without them 11 tests fail, not the 8 usually
quoted.

**Where `PARTITION_GATES.md` lives:** it is on `hexagon-mergeq` and
`hexagon-integration-a47`, and NOT on `hexagon-gateinv` alone — the gate inventory's
own doc came in on this side of the merge. Correcting the brief here: §36.1 said the
file "is not in a47"; it is. What is not in a47 is the *model corpus* it needs (§9).
