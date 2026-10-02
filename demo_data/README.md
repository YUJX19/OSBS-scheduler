# Demonstration subset of the UAV-to-UAV sidelink corpora

A small cut of the two released corpora, shipped with the inference code for trying the trained OSBS models without downloading the full dataset: 64 test-split episodes of the General corpus and 64 of the ray-traced Leipzig urban-demo corpus, stored under the historical `futian` key, each with its causal observation, its stored future and its budget labels of all five budgets, in the same file format as the full corpora (same datasets, dtypes, compression and `unit` / `description` attributes), so that the inference code reads either file unchanged.

```
general/observations.h5     what the scheduler sees at the decision subframe (64 episodes of otfs_u2u_general_main, test split)
general/targets.h5          the stored future: clean per-bin SINR of the 8 repetition subframes of each of the 12 windows, the per-bin repetition map (smallest K at which the bin's Chase-combined clean SINR reaches the threshold of the protocol MCS, 9 = unreachable within 8), the per-MCS thresholds
general/budget_labels.h5    the class of every bin of the repetition map (0 for bins that need K = 1, 1 for K = 2, 2 for K = 3, 4, 5, 6, 7, 8, 3 for bins unreachable within 8) and, for each budget 512, 1024, 1536, 2048 and 4096, the window's feasibility, the label's K per class, usage and utility
futian/...                  the same three files for 64 episodes of otfs_u2u_futian_main, the historical identifier for the Leipzig urban-demo corpus (test split, the held-out site quadrant)
manifest.json               source episode ids, SHA-256 and size of every file, the fields of every file
```

The observations of an episode do not depend on the budget; the budget enters through the labels and through the repetition rule of a checkpoint.  Episodes are indexed 0 to 63 inside the files; the full-corpus episodes 3239 and 15947 used as the example in the inference README are indices 11 and 53 of the General file.  Every file also holds `source_episode_id`, the index of the episode in the full corpus.

```
python3 predict.py                        # the defaults: this General file, episodes 11 and 53, scored on the stored future
python3 predict.py --corpus demo_data/futian --checkpoint checkpoints/futian/fine_tuned/B1024/film/best.pt --episodes 0 1
```
