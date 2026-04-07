---
name: batch-processing
description: >
  Process large datasets that exceed context-window capacity. Use when handling
  changelogs, migrations, audits, or any task requiring systematic processing of
  many items.
---

## Core principle

When the dataset is larger than your working memory, the strategy must be
decided **before** you start reading — not discovered halfway through when items
get lost.

**Map-reduce over files.** Each batch writes to its own output file (map). The
final result is produced by reading those per-batch outputs (reduce).

## When to trigger

Activate when:

-   The number of items to process exceeds ~50.
-   The raw data exceeds ~800 lines (one view_file window).
-   A task naturally decomposes into "process each item, then aggregate."

Examples: changelogs, CL audits, migration lists, bulk code reviews,
dependency analyses.

## Variant: interactive multi-session review

When items need **human decisions** across multiple sessions (not just
automated processing), co-locate all state with a skill:

1.  **Tracking JSON** — categories like `unsorted`, `validated`,
    `excluded_certain`, `excluded_maybe`. All items start as `unsorted`.
2.  **Validation script** — ensures completeness (all items on disk are
    tracked), no duplicates, no ghosts. Run before and after each session.
3.  **Skill file** — documents review criteria, steps, and categories.
    Merges the workflow into the skill (skills are auto-discoverable;
    workflows require knowing the filename).
4.  **Group by category, not alphabetically** — reviewing related items
    together produces better decisions. Categorize first, then present
    one category per batch.

Reference implementation: `review-skills/`.

## Procedure

### 1. Estimate scope

Before reading any data, establish the total count. Use commands like
`wc -l`, `g4 changes ... | wc -l`, or equivalent to get an exact number.
This number becomes the **completeness target**.

### 2. Dump to a persistent file

Write the full raw data to a file in `experimental/`. This decouples data acquisition from processing — you can re-read
any batch without re-running the query.

### 3. Design the batch plan

Decide upfront:

-   **Batch size**: how many lines/items per pass (typically ~50 elements, or
    ~800 lines for `view_file`).
-   **Number of batches**: `ceil(total / batch_size)`.
-   **Enumerate all batches in `task.md`**: list every batch with its exact line
    range or IDs. This is the checklist — a batch is only done when its checkbox
    is ticked.
-   **Write ALL batch checkboxes in `task.md` now.**
-   **Add a `refresh-context` checkbox in `task.md`.**: During long tasks,
    context might be forgotten. This forces you to re-cover the relevant
    context after batches are processed.
-   **Per-batch output path**: each batch writes to its own file (e.g.
    `batch_01.md`, `batch_02.md`). Decide the naming scheme and format now.

### 4. Map: process each batch

For each batch:

1.  Read the batch (e.g. `view_file` with `StartLine` / `EndLine`).
2.  Process all items in the batch.
3.  **Write** results to that batch's **own output file** — not a shared file,
    not memory. This is the critical invariant: without per-batch files,
    `refresh-context` has nothing to reload after truncation, and the reduce
    phase degrades to writing from a lossy checkpoint summary.
4.  Update `task.md` with the batch status (lines covered, items processed).
5.  Follow `refresh-context` after a checkpoint summary or every ~3 batches.

### 5. Verify completeness (hard gate)

**You must not proceed to the reduce step until this passes.**

1.  Walk the batch checklist in `task.md`. Every batch must be checked off. If
    any is missing, go back and process it.
2.  Verify that `last_batch_end_line >= total_lines`. If not, batches are
    missing.
3.  Count processed items against the scope estimate from step 1.
4.  If counts don't match, identify the gap (dropped items, skipped batches).

### 6. Reduce: produce the final output

In long tasks, everything loaded at the start (skills, batch files) gets
truncated from context before this phase. Follow `refresh-context`, and apply it to reload:

1.  **Re-read all relevant skills** — not just the primary one. Re-read every
    skill that governs the output format, style, or verification procedure.
2.  **Re-read all batch output files** to bring them into context. Don't rely on
    memory of what they contained.
3.  Aggregate into the final result (changelog, report, summary).

## Anti-patterns

-   ❌ Starting to process before knowing the total count.
-   ❌ Writing batch results to a single shared file (loses batch isolation;
    makes reruns destructive).
-   ❌ Holding intermediate results in memory instead of writing them to a file
    after each batch.
-   ❌ Declaring "done" without a completeness check against the original count.
-   ❌ Producing aggregate output before all batches are processed — even if the
    remaining batches "look" unimportant.
-   ❌ Listing `batch-processing` in `task.md` without executing its steps —
    referencing a skill is not the same as following it.

## Summarization

Instructions for the summarizer agent: When `summarizing this work`, do NOT
copy the extracted informations. They are saved to the intermediate batch files
already. Instead instruct the model to re-read the batch it saved (using
`refresh-context`).
