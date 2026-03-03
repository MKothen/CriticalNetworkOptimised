"""
Statistical testing functions
Compares experimental conditions using appropriate parametric/non-parametric tests
"""

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def _run_posthoc_tukey(valid_groups, valid_labels):
    """Run Tukey HSD post-hoc test and print results."""
    all_data_flat = np.concatenate(valid_groups)
    labels_flat = np.concatenate([
        np.full(len(group), label) for label, group in zip(valid_labels, valid_groups)
    ])
    tukey_results = pairwise_tukeyhsd(all_data_flat, labels_flat, alpha=0.05)
    print(tukey_results)


def _run_posthoc_mannwhitney(valid_groups, valid_labels):
    """Run Mann-Whitney U post-hoc tests with Bonferroni correction."""
    pairs = list(combinations(range(len(valid_groups)), 2))
    corrected_alpha = 0.05 / len(pairs)
    print(f"  - Bonferroni corrected significance level (alpha): {corrected_alpha:.4f}")

    for i, j in pairs:
        u_stat, p_val_mw = stats.mannwhitneyu(
            valid_groups[i], valid_groups[j], alternative="two-sided"
        )
        sig_marker = '**SIGNIFICANT**' if p_val_mw < corrected_alpha else ''
        print(f"  - Comparing '{valid_labels[i]}' vs '{valid_labels[j]}': "
              f"p = {p_val_mw:.4f} {sig_marker}")


def _check_assumptions(valid_groups, valid_labels):
    """
    Check normality and equal variance assumptions.

    Returns
    -------
    bool
        True if assumptions are met for parametric tests
    """
    print("\n(1) Checking assumptions for parametric tests (ANOVA)...")
    is_normal = True
    for i, group in enumerate(valid_groups):
        shapiro_stat, shapiro_p = stats.shapiro(group)
        if shapiro_p < 0.05:
            is_normal = False
        status = 'Not Normal' if shapiro_p < 0.05 else 'Normal'
        print(f"  - Normality (Shapiro-Wilk) for group '{valid_labels[i]}': "
              f"p = {shapiro_p:.4f} ({status})")

    levene_stat, levene_p = stats.levene(*valid_groups)
    has_equal_variances = levene_p >= 0.05
    status = 'Variances are Equal' if has_equal_variances else 'Variances are Unequal'
    print(f"  - Homogeneity of Variances (Levene's Test): p = {levene_p:.4f} ({status})")

    assumptions_met = is_normal and has_equal_variances
    print(f"\nConclusion: Assumptions for ANOVA are {'MET.' if assumptions_met else 'NOT MET.'}")
    return assumptions_met


def run_and_print_statistical_tests(results_dict, ei_ratios, imid_vals, num_repetitions, condition_map):
    """
    Perform statistical tests comparing conditions.

    Parameters
    ----------
    results_dict : dict
        Dictionary of metric arrays with shape (len(imid), len(ei_ratios), num_repetitions)
    ei_ratios : array
        E/I ratio values tested
    imid_vals : array
        Input current values tested
    num_repetitions : int
        Number of repetitions per condition
    condition_map : dict
        Mapping from E/I ratios to condition names
    """
    if num_repetitions < 3:
        print("--- SKIPPING STATISTICAL TESTS: Need at least 3 repetitions. ---")
        return

    print("=" * 80)
    print(" " * 20 + "STATISTICAL ANALYSIS OF RESULTS")
    print("=" * 80)

    for metric_name, data_array in results_dict.items():
        print(f"\n--- Analysis for Metric: {metric_name} ---")

        for i_idx, imid in enumerate(imid_vals):
            print(f"\n--- Condition: Imid = {imid:.4f} nA ---")

            # Build valid groups (filter NaN, require >= 3 samples)
            valid_groups = []
            valid_labels = []
            for k in range(len(ei_ratios)):
                group = data_array[i_idx, k, :]
                clean = group[~np.isnan(group)]
                if len(clean) >= 3:
                    valid_groups.append(clean)
                    valid_labels.append(
                        condition_map.get(ei_ratios[k], f"EI_{ei_ratios[k]:.3f}").capitalize()
                    )

            if len(valid_groups) < 2:
                print("Could not perform test: fewer than two valid groups with sufficient data.")
                continue

            assumptions_met = _check_assumptions(valid_groups, valid_labels)

            print("\n(2) Performing appropriate statistical test...")

            if assumptions_met:
                print("\nUsing parametric test: One-Way ANOVA")
                f_stat, p_val = stats.f_oneway(*valid_groups)
                print(f"  - F-statistic: {f_stat:.4f}")
                print(f"  - P-value: {p_val:.4f}")

                if p_val < 0.05:
                    print("  - Result: **Significant difference detected** (p < 0.05).")
                    print("\n  Post-Hoc Test: Tukey's HSD")
                    _run_posthoc_tukey(valid_groups, valid_labels)
                else:
                    print("  - Result: No significant overall difference detected (p >= 0.05).")
            else:
                print("\nUsing non-parametric test: Kruskal-Wallis H-test")
                h_stat, p_val = stats.kruskal(*valid_groups)
                print(f"  - H-statistic: {h_stat:.4f}")
                print(f"  - P-value: {p_val:.4f}")

                if p_val < 0.05:
                    print("  - Result: **Significant difference detected** (p < 0.05).")
                    print("\n  Post-Hoc Tests: Mann-Whitney U with Bonferroni Correction")
                    _run_posthoc_mannwhitney(valid_groups, valid_labels)
                else:
                    print("  - Result: No significant overall difference detected (p >= 0.05).")

            print("=" * 80)

    print(" " * 24 + "END OF STATISTICAL ANALYSIS")
    print("=" * 80)


def run_learning_curve_statistics(all_learning_data, imid_param_values, ei_ratio_param_values, condition_map):
    """
    Performs statistical tests at each measurement point of the learning curves.

    Parameters
    ----------
    all_learning_data : array
        Learning curve data with shape (len(imid), len(ei_ratio), num_repetitions)
    imid_param_values : array
        Imid values tested
    ei_ratio_param_values : array
        E/I ratio values tested
    condition_map : dict
        Mapping from E/I ratios to condition names
    """
    print("=" * 80)
    print(" " * 15 + "STATISTICAL ANALYSIS OF LEARNING CURVES")
    print("=" * 80)

    # Build records efficiently
    records = []
    for i_idx, imid_val in enumerate(imid_param_values):
        for j_idx, ei_val in enumerate(ei_ratio_param_values):
            condition_name = condition_map.get(ei_val, f"EI_{ei_val:.3f}").capitalize()
            for rep_idx in range(len(all_learning_data[i_idx][j_idx])):
                lc = all_learning_data[i_idx][j_idx][rep_idx]
                if lc and isinstance(lc, dict):
                    for n_samples, accuracy in lc.items():
                        records.append((imid_val, ei_val, condition_name, rep_idx, n_samples, accuracy))

    if not records:
        print("No learning curve data available for statistical analysis.")
        return

    df = pd.DataFrame(records,
                      columns=["Imid", "EI_Ratio", "Condition", "Repetition",
                               "Training_Samples", "Accuracy"])

    for size in sorted(df["Training_Samples"].unique()):
        print(f"\n--- Analysis at Training Size: {size} Samples ---")

        df_size = df[df["Training_Samples"] == size]

        conditions = df_size["Condition"].unique()
        groups_data = [df_size.loc[df_size["Condition"] == c, "Accuracy"].values for c in conditions]

        valid_groups = [g for g in groups_data if len(g) >= 3]
        valid_labels = [conditions[i] for i, g in enumerate(groups_data) if len(g) >= 3]

        if len(valid_groups) < 2:
            print("  Cannot perform test: fewer than two conditions with sufficient data.")
            continue

        f_stat, p_val = stats.f_oneway(*valid_groups)
        print(f"  One-Way ANOVA result: F-statistic = {f_stat:.4f}, p-value = {p_val:.4f}")

        if p_val < 0.05:
            print("  ANOVA is significant. Performing Tukey's HSD post-hoc test...")
            _run_posthoc_tukey(valid_groups, valid_labels)
        else:
            print("  No significant overall difference detected at this sample size.")

    print("=" * 80)
    print(" " * 22 + "END OF LEARNING CURVE ANALYSIS")
    print("=" * 80)
