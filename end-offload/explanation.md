# Why promote restores less than 100% of the footprint

## Summary

In the end-of-task experiment (`replay_end.py`, 100 tasks), the container's memory after promote is a
median **74%** of its memory at the freeze (range 61–91%). The gap is a median **24 MiB** per task.

**Nothing is lost.** Promote brings back all process memory and all cached file contents. The gap is
**clean kernel cache that the kernel rebuilds on its own when it is needed**:

- **dentry and inode caches** (kernel memory): measured directly;
- **filesystem metadata blocks** (directory blocks, inode tables): identified by elimination;
- **container-runtime files** (runc and its libraries): a constant ~0.2 MiB.

Promote does not restore these because it only restores what it recorded: process pages and the contents
of cached files. These caches are not part of any recorded file, so there is nothing to read them back from.

## What the recovered fraction measures

```
promote_recovered_fraction = memory.usage_in_bytes after promote / memory.usage_in_bytes at the freeze
```

`memory.usage_in_bytes` is everything charged to the container's memory cgroup:

| Charged to the cgroup | Example |
|---|---|
| process (anonymous) memory | heap and stack of live processes; at the end of a task only `sleep`, ~0.09 MiB |
| page cache of files | contents of Python files, libraries and data the task read or wrote |
| filesystem metadata blocks | ext4 directory blocks and inode tables (counted as cache) |
| kernel memory | dentries, inodes, other slab objects |

At the end of a task almost all of it is page cache (median 89 of 101 MiB) plus kernel memory (median 12.6 MiB).

## What demote and promote do

**Demote** (container frozen):
1. fsync the container's writable layer, so every written byte is on disk;
2. record every cached page of every file under the container's filesystem (a `mincore` scan) and copy
   those pages to a snapshot file;
3. squeeze the cgroup's memory limit to ~0: process memory goes to swap, and the kernel drops everything
   clean it can: file pages, metadata blocks, dentries, inodes.

**Promote** (container still frozen):
1. lift the limit;
2. a helper process inside the container's cgroup reads back every recorded process page and every
   recorded file range.

Promote's plan contains only **process pages** and **file ranges**. Anything else that demote dropped stays
dropped.

## Why the three caches are not restored

| Memory | Demote | Promote | Back? |
|---|---|---|---|
| process memory | swapped out | read back from the recorded page list | yes |
| file contents | dropped, saved in the snapshot | recorded ranges re-read | yes |
| dentries and inodes | freed (reclaimable slab) | not in the plan | partly |
| metadata blocks | dropped (clean) | not in the plan | partly |
| runc and its libraries | dropped (clean) | not in the plan: outside the container's files | no |

1. **They belong to no file in the plan.** A dentry or inode is a kernel object created by a path lookup or
   a `stat`. A directory block or inode-table block belongs to the disk device, not to any file in the
   container. Promote restores by re-reading file ranges, so there is nothing to re-read for them.
2. **Demote drops them first because they are clean.** Nothing has to be written, since the kernel can always
   rebuild them from disk.
3. **Nothing rebuilds them during promote.** They normally come back as processes look up paths, and the
   container is frozen.

They come back **partly** as a side effect: when promote opens a file to re-read it, the path lookup recreates
that file's dentry, inode and a few metadata blocks (median kernel memory 12.6 MiB at the freeze, 6.3 MiB after
promote). Paths the task only listed or stat-ed are not re-opened, so their caches stay gone.

Some kernel memory is never removed at all: the squeeze cannot free it (median residual 6.7 MiB after demote,
of which 6.4 MiB is kernel memory). That part stays resident, so it does not contribute to the gap.

## Evidence

### Nothing is lost (main experiment, 100 tasks)

- 0 read errors in promote, in every task;
- 0 bytes left in swap after promote, in every task;
- at most 22 cached pages (median 2) still resident after demote, so demote did remove the footprint;
- every written file was fsynced before demote.

### What the gap is made of (`recovery_gap.py`)

`recovery_gap.py` runs small workloads, one fresh container each, demotes and promotes them with the same
`measure_offload()` as the main experiment, and attributes every cached page to an owner with
`/proc/kpagecgroup`, at the freeze and after promote. It reproduces the main experiment: the replay workload
recovers **0.92** for astropy-13033 (main experiment: 0.90) and **0.61** for django-11603 (main: 0.61).

MiB lost between the freeze and after promote (astropy-13033 / django-11603):

| Workload | What it does | Lost | File contents | Kernel memory | "other" (metadata) | runc |
|---|---|---|---|---|---|---|
| idle | nothing | 0.9 / 1.1 | 0 | 0.1 / 0.3 | 0.4 / 0.6 | 0.2 |
| exec40 | 40 × `docker exec true` | 5.2 / 5.6 | 0 | 1.1 / 1.1 | 3.1 / 3.5 | 0.2 |
| find_names | list every directory | 45.5 / 58.0 | 0 | 2.9 / 3.9 | **42 / 54** | 0.2 |
| find_stat | list and stat every file | 172 / 169 | 0 | **129 / 117** | 43 / 52 | 0.2 |
| cat_testbed | read every file in `/testbed` | 8.8 / 35 | 0 | 3.6 / 15 | 4.7 / 19 | 0.2 |
| copyup | read, then edit 50 files | 6.9 / 6.7 | 0 | 2.0 / 1.9 | 3.6 / 4.1 | 0.2 |
| **replay** | the task's real tool calls | **10.9 / 17.2** | **0** | 3.4 / 1.4 | **6.3 / 15.2** | 0.2 |

How to read it:

- **File contents: 0 lost in every workload.** The container's cached file pages are identical before and after.
- **Kernel memory is dentries and inodes.** `find_stat` adds ~120 MiB, and the host's slab caches grow by the same
  amount (+50 MiB overlay inodes, +18–34 MiB ext4 inodes, +27–28 MiB dentries).
- **"other" is created by walking directories.** Listing directories alone (`find_names`) creates 42–54 MiB of it,
  and stat-ing on top adds almost nothing. It is larger for Django, whose tree has more files.
- **runc is a constant 0.2 MiB.** Copy-up of edited files is negligible (the 50 originals are 0.37 / 0.09 MiB).

### "other" is filesystem metadata (by elimination)

The per-container scan of the disk device (`/dev/md0`) could not see block-device cache, so metadata cannot be
attributed to the container page by page. The evidence is indirect:

- "other" appears from directory traversal without any file being read;
- a host-side check (`find` over 450k entries of Docker's storage) grew the kernel's block-device cache
  (`Buffers` in `/proc/meminfo`) from 4.4 to 760 MiB, about 1.7 KB per entry. Traversal does fill exactly this cache;
- everything else charged as cache is accounted for: file contents (attributed page by page) and runc.

## Why Django recovers less than astropy

Median recovery is **0.90 for astropy** (22 tasks) and **0.73 for Django** (78 tasks). Django tasks touch many
more small files (median 2,448 cached files against 1,278), so they build more dentries, inodes and metadata per MiB of file
content. In the Django replay, metadata is 15 of the 17 lost MiB.

## What this means

- Promote restores everything the container's processes use as data: their memory and their file contents.
- The unrecovered part is cache the kernel rebuilds on demand. After the container is thawed, the first lookup of
  each path it had cached costs a small disk read; the files and their contents are the same.
- Reaching 100% would require replaying the traversal (for example, a `stat` of every path the task touched).
  Nothing records which paths were cached, and the replay would add time to promote for caches that rebuild anyway.
- The plotted offload size ("swap + cache snapshot") excludes these caches; `freeze_usage_in_bytes` includes them.

## Limits of this explanation

- The breakdown was measured on 2 tasks. The other 98 tasks show the same signature (no read errors, no residual
  swap, lower recovery when more files are touched) but were not broken down.
- The metadata share is inferred, not measured per container. Recording the host's `Buffers` growth during each
  workload on a quiet machine would measure it directly.
- Slab growth is measured host-wide (the per-cgroup breakdown is empty on this kernel), so other activity on the
  machine adds noise to it.

## Reproducing

From `end-offload/`, as root, with swap on and no other experiment running on the machine:

```
sudo ../.venv/bin/python recovery_gap.py ../../swebench-runs/qwen-run-20260923T053154Z-PF12hU astropy__astropy-13033
sudo ../.venv/bin/python recovery_gap.py ../../swebench-runs/qwen-run-20260923T073021Z-LtPzSb django__django-11603
```

Results go to `results-gap/<task>/<workload>.json`. Each run takes ~20 minutes, mostly the disk-device scan.

## Comparison with the peak experiment

Both experiments count the container's memory the same way: everything charged to its memory cgroup, including
kernel memory and filesystem metadata. Medians over the same 100 tasks:

| | Peak (frozen early in the task) | End (frozen when the task is done) |
|---|---|---|
| container memory at the freeze | 17.6 MiB | 101.0 MiB |
| process memory | 1.75 MiB (max 62) | 0.09 MiB (max 0.09) |
| page cache (file contents + metadata) | 13.9 MiB | 89.0 MiB |
| kernel memory | 1.4 MiB | 12.6 MiB |
| transferred to disk and back | 13.9 MiB | 83.5 MiB |
| demote time | 69 ms | 466 ms |
| promote time | 29 ms | 306 ms |
| recovered | 0.85 | 0.74 |
| not recovered | 2.6 MiB | 24.3 MiB |
| recovered, astropy / Django | 0.91 / 0.85 | 0.89 / 0.73 |

What the comparison shows:

1. **A finished task holds about 6× more memory than it does early on.** Over the task the container accumulates
   page cache (files read and written) and kernel memory (paths looked up), and none of it is released when the
   tool calls exit.
2. **The make-up changes.** Early in a task, live processes hold memory (up to 62 MiB). At the end, process memory is
   only the container's idle `sleep`; the footprint is almost entirely page cache plus kernel memory.
3. **Offload takes about 7–10× longer at the end,** in line with the larger amount to transfer
   (83.5 vs 13.9 MiB). Demote stays under 1.2 s and promote under 1.2 s in every task.
4. **Recovery is lower at the end,** for the reason explained above: the longer a task runs, the more paths it has
   looked up, and the larger the share of its footprint that is kernel memory and metadata, which promote does
   not restore. The not-recovered amount grows from 2.6 to 24.3 MiB, about 9× while the footprint grows about 6×.
5. **The gap between astropy and Django appears only at the end.** Early on both recover about the same
   (0.91 / 0.85). By the end, Django tasks have touched many more small files and drop to 0.73, while astropy
   stays at 0.89.
6. **No data is lost in either experiment.** In both, what promote does not restore is clean kernel cache, and the
   file contents and process memory come back.
