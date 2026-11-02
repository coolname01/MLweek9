# Data checks

Updated: 2026-11-02. Status: **1 discrepancy listed**.

One NPZ at a time (`np.load`). `team_info.csv` was not opened. Official record and epoch counts were not changed to fit the files.

Runtime: 58.0 s. Python 3.14.8, numpy 2.5.4, pandas 3.1.0, pyyaml 6.0.3. Windows-11-10.0.26300-SP0.
Seed in config: 42. Signal contract: fs=100, epoch_length=3000, channels=['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal'].

## Counts

| split | records | expected records | epochs | expected epochs | status |
| --- | --- | --- | --- | --- | --- |
| train | 112 | 112 | 302653 | 302653 | ok |
| val | 16 | 16 | 43206 | 43206 | ok |
| test | 24 | 24 | 66222 | 66222 | ok |

## Subjects

Unique `subject_id`: train 58, val 8, overlap 0. Identifiers are not listed.

## Class distribution

Shares are within the split, in percent. Labels come from `labels.csv`.

| split | class | label | epochs | share % |
| --- | --- | --- | --- | --- |
| train | W | 0 | 208224 | 68.80 |
| train | N1 | 1 | 16277 | 5.38 |
| train | N2 | 2 | 51233 | 16.93 |
| train | N3 | 3 | 8412 | 2.78 |
| train | REM | 4 | 18507 | 6.11 |
| val | W | 0 | 28763 | 66.57 |
| val | N1 | 1 | 1884 | 4.36 |
| val | N2 | 2 | 7459 | 17.26 |
| val | N3 | 3 | 1784 | 4.13 |
| val | REM | 4 | 3316 | 7.67 |

## epoch_index gaps

A gap is a forward step greater than 1. Length is the number of missing 30-second slots (`step - 1`). Lengths are listed in file order.

| split | record_id | epochs | n_gaps | missing slots | gap lengths |
| --- | --- | --- | --- | --- | --- |
| train | SC4011 | 2802 | 0 | 0 |  |
| train | SC4012 | 2848 | 0 | 0 |  |
| train | SC4021 | 2804 | 0 | 0 |  |
| train | SC4022 | 2755 | 1 | 1 | 1 |
| train | SC4031 | 2820 | 0 | 0 |  |
| train | SC4032 | 2732 | 0 | 0 |  |
| train | SC4051 | 2722 | 0 | 0 |  |
| train | SC4052 | 2804 | 2 | 2 | 1 1 |
| train | SC4061 | 2770 | 0 | 0 |  |
| train | SC4062 | 2830 | 0 | 0 |  |
| train | SC4081 | 2796 | 0 | 0 |  |
| train | SC4082 | 2634 | 0 | 0 |  |
| train | SC4091 | 2721 | 10 | 11 | 1 2 1 1 1 1 1 1 1 1 |
| train | SC4092 | 2044 | 13 | 814 | 1 1 1 1 1 1 1 1 1 1 1 1 802 |
| train | SC4101 | 2719 | 1 | 1 | 1 |
| train | SC4102 | 2857 | 1 | 1 | 1 |
| train | SC4111 | 2641 | 1 | 1 | 1 |
| train | SC4112 | 2780 | 0 | 0 |  |
| train | SC4121 | 2685 | 0 | 0 |  |
| train | SC4122 | 2606 | 0 | 0 |  |
| train | SC4131 | 2814 | 0 | 0 |  |
| train | SC4141 | 2756 | 0 | 0 |  |
| train | SC4142 | 2774 | 0 | 0 |  |
| train | SC4151 | 2616 | 4 | 4 | 1 1 1 1 |
| train | SC4152 | 2859 | 5 | 5 | 1 1 1 1 1 |
| train | SC4161 | 2621 | 4 | 5 | 1 2 1 1 |
| train | SC4162 | 2750 | 2 | 2 | 1 1 |
| train | SC4191 | 2774 | 0 | 0 |  |
| train | SC4192 | 2605 | 4 | 5 | 2 1 1 1 |
| train | SC4201 | 2803 | 1 | 1 | 1 |
| train | SC4202 | 2670 | 0 | 0 |  |
| train | SC4221 | 2700 | 0 | 0 |  |
| train | SC4222 | 2760 | 0 | 0 |  |
| train | SC4231 | 2743 | 3 | 3 | 1 1 1 |
| train | SC4232 | 2635 | 5 | 5 | 1 1 1 1 1 |
| train | SC4291 | 2756 | 0 | 0 |  |
| train | SC4292 | 2807 | 3 | 3 | 1 1 1 |
| train | SC4301 | 2644 | 2 | 2 | 1 1 |
| train | SC4302 | 2808 | 0 | 0 |  |
| train | SC4311 | 2670 | 0 | 0 |  |
| train | SC4312 | 2700 | 0 | 0 |  |
| train | SC4321 | 2689 | 1 | 1 | 1 |
| train | SC4322 | 2616 | 0 | 0 |  |
| train | SC4341 | 2750 | 0 | 0 |  |
| train | SC4342 | 2790 | 0 | 0 |  |
| train | SC4351 | 2710 | 0 | 0 |  |
| train | SC4352 | 2530 | 2 | 2 | 1 1 |
| train | SC4362 | 2266 | 0 | 0 |  |
| train | SC4381 | 2750 | 0 | 0 |  |
| train | SC4382 | 2770 | 0 | 0 |  |
| train | SC4401 | 2630 | 0 | 0 |  |
| train | SC4402 | 2790 | 0 | 0 |  |
| train | SC4421 | 2764 | 0 | 0 |  |
| train | SC4422 | 2682 | 0 | 0 |  |
| train | SC4441 | 2620 | 0 | 0 |  |
| train | SC4442 | 2778 | 0 | 0 |  |
| train | SC4451 | 2790 | 0 | 0 |  |
| train | SC4452 | 2670 | 0 | 0 |  |
| train | SC4461 | 2695 | 1 | 1 | 1 |
| train | SC4462 | 2854 | 2 | 2 | 1 1 |
| train | SC4471 | 2739 | 1 | 1 | 1 |
| train | SC4472 | 2789 | 1 | 1 | 1 |
| train | SC4482 | 2880 | 0 | 0 |  |
| train | SC4491 | 2800 | 0 | 0 |  |
| train | SC4492 | 2135 | 1 | 1 | 1 |
| train | SC4501 | 2750 | 0 | 0 |  |
| train | SC4502 | 2770 | 0 | 0 |  |
| train | SC4522 | 2774 | 0 | 0 |  |
| train | SC4531 | 2602 | 0 | 0 |  |
| train | SC4532 | 2744 | 0 | 0 |  |
| train | SC4541 | 2739 | 1 | 1 | 1 |
| train | SC4542 | 2680 | 0 | 0 |  |
| train | SC4551 | 2772 | 0 | 0 |  |
| train | SC4552 | 2810 | 0 | 0 |  |
| train | SC4561 | 2702 | 2 | 2 | 1 1 |
| train | SC4562 | 2800 | 0 | 0 |  |
| train | SC4581 | 2738 | 0 | 0 |  |
| train | SC4582 | 2634 | 0 | 0 |  |
| train | SC4621 | 2612 | 0 | 0 |  |
| train | SC4622 | 2856 | 0 | 0 |  |
| train | SC4631 | 2758 | 0 | 0 |  |
| train | SC4632 | 2848 | 0 | 0 |  |
| train | SC4641 | 2680 | 0 | 0 |  |
| train | SC4642 | 2790 | 0 | 0 |  |
| train | SC4651 | 2860 | 0 | 0 |  |
| train | SC4652 | 2838 | 1 | 2 | 2 |
| train | SC4661 | 2664 | 0 | 0 |  |
| train | SC4662 | 2820 | 0 | 0 |  |
| train | SC4671 | 2780 | 0 | 0 |  |
| train | SC4672 | 2580 | 0 | 0 |  |
| train | SC4701 | 2679 | 1 | 1 | 1 |
| train | SC4702 | 2624 | 0 | 0 |  |
| train | SC4711 | 2730 | 0 | 0 |  |
| train | SC4712 | 2820 | 0 | 0 |  |
| train | SC4721 | 2342 | 0 | 0 |  |
| train | SC4722 | 2746 | 0 | 0 |  |
| train | SC4731 | 2780 | 0 | 0 |  |
| train | SC4732 | 2519 | 1 | 1 | 1 |
| train | SC4741 | 2689 | 1 | 1 | 1 |
| train | SC4742 | 2650 | 0 | 0 |  |
| train | SC4751 | 2680 | 0 | 0 |  |
| train | SC4752 | 2470 | 0 | 0 |  |
| train | SC4761 | 2610 | 2 | 20 | 16 4 |
| train | SC4762 | 2662 | 26 | 148 | 1 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 97 15 10 1 |
| train | SC4771 | 2756 | 0 | 0 |  |
| train | SC4772 | 2242 | 0 | 0 |  |
| train | SC4801 | 2774 | 0 | 0 |  |
| train | SC4802 | 2810 | 0 | 0 |  |
| train | SC4811 | 2402 | 0 | 0 |  |
| train | SC4812 | 2414 | 2 | 2 | 1 1 |
| train | SC4821 | 2721 | 1 | 1 | 1 |
| train | SC4822 | 2810 | 0 | 0 |  |
| val | SC4041 | 2569 | 1 | 1 | 1 |
| val | SC4042 | 2788 | 4 | 4 | 1 1 1 1 |
| val | SC4071 | 2810 | 0 | 0 |  |
| val | SC4072 | 2770 | 0 | 0 |  |
| val | SC4171 | 2741 | 1 | 1 | 1 |
| val | SC4172 | 2720 | 3 | 4 | 1 2 1 |
| val | SC4241 | 2700 | 2 | 2 | 1 1 |
| val | SC4242 | 2710 | 0 | 0 |  |
| val | SC4261 | 2800 | 0 | 0 |  |
| val | SC4262 | 2720 | 0 | 0 |  |
| val | SC4271 | 2440 | 0 | 0 |  |
| val | SC4272 | 2870 | 0 | 0 |  |
| val | SC4571 | 2836 | 1 | 14 | 14 |
| val | SC4572 | 2873 | 7 | 7 | 1 1 1 1 1 1 1 |
| val | SC4591 | 2819 | 1 | 1 | 1 |
| val | SC4592 | 2040 | 0 | 0 |  |
| test | test_001 | 2728 | 0 | 0 |  |
| test | test_002 | 2750 | 0 | 0 |  |
| test | test_003 | 2840 | 0 | 0 |  |
| test | test_004 | 2760 | 0 | 0 |  |
| test | test_005 | 2765 | 0 | 0 |  |
| test | test_006 | 2730 | 0 | 0 |  |
| test | test_007 | 2814 | 0 | 0 |  |
| test | test_008 | 2850 | 0 | 0 |  |
| test | test_009 | 2730 | 0 | 0 |  |
| test | test_010 | 2809 | 1 | 1 | 1 |
| test | test_011 | 2780 | 0 | 0 |  |
| test | test_012 | 2805 | 5 | 5 | 1 1 1 1 1 |
| test | test_013 | 2665 | 1 | 1 | 1 |
| test | test_014 | 2712 | 0 | 0 |  |
| test | test_015 | 2758 | 2 | 2 | 1 1 |
| test | test_016 | 2842 | 0 | 0 |  |
| test | test_017 | 2694 | 0 | 0 |  |
| test | test_018 | 2788 | 0 | 0 |  |
| test | test_019 | 2756 | 0 | 0 |  |
| test | test_020 | 2723 | 0 | 0 |  |
| test | test_021 | 2800 | 0 | 0 |  |
| test | test_022 | 2644 | 3 | 6 | 2 3 1 |
| test | test_023 | 2650 | 0 | 0 |  |
| test | test_024 | 2829 | 1 | 1 | 1 |

## Manifest

Compared: 162. Bytes and SHA-256 matched: 161. Skipped unread `team_info.csv`: 1.

## Discrepancies

- manifest file missing locally: README.md (looked at archive/README.md, manifest bytes 2307)
