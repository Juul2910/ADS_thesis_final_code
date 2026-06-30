"""
Small EC sanity check.

Goal:
Create a tiny artificial test where the summary focuses only on one subgroup,
so Equal Coverage should become noticeably higher than a balanced summary.
"""

import pandas as pd

from compute_metrics import (
    compute_equal_coverage,
    get_group_label,
    get_document_texts,
)

# --------------------------------------------------
# 1. Tiny test dataset
# --------------------------------------------------

df = pd.DataFrame([
    {
        "Feedback": "A woman requested food and hygiene support after a stroke.",
        "sex": "women's",
        "region": "Region A",
    },
    {
        "Feedback": "A woman asked for information about starting a small business.",
        "sex": "women's",
        "region": "Region A",
    },
    {
        "Feedback": "A man asked about employment vacancies.",
        "sex": "men's",
        "region": "Region B",
    },
    {
        "Feedback": "A man needed help with passport documentation.",
        "sex": "men's",
        "region": "Region B",
    },
])

entries = df.to_dict(orient="records")

# make keys lowercase, matching your compute_metrics.py style
entries = [
    {k.lower(): str(v) for k, v in row.items()}
    for row in entries
]

doc_texts = get_document_texts(entries)

# --------------------------------------------------
# 2. Biased summary: only focuses on women's entries
# --------------------------------------------------

biased_summary = """
Women in the feedback requested food, hygiene support, and information about starting a small business.
The main concerns are support after health problems and livelihood assistance for women.
"""

doc_labels = [
    get_group_label(entry, "sex", [])
    for entry in entries
]

print("Document labels:")
print(doc_labels)

print("\nDocument texts:")
for t in doc_texts:
    print("-", t)

# --------------------------------------------------
# 3. Compute EC for sex
# --------------------------------------------------

ec_result = compute_equal_coverage(
    doc_texts=doc_texts,
    doc_labels=doc_labels,
    summary=biased_summary,
)

print("\n=== EC TEST RESULT ===")
print("Biased summary:")
print(biased_summary)

print("\nEC(sex):", ec_result["ec_score"])
print("Overall coverage:", ec_result["overall_coverage_prob"])
print("Per-group coverage:", ec_result["per_group_coverage"])
print("Group counts:", ec_result["group_counts"])

# --------------------------------------------------
# 4. Perfect-match EC sanity check
# --------------------------------------------------
# Goal:
# Check what happens when the summary exactly matches the feedback.
#
# Expected outcome:
# - coverage should be high
# - EC should be 0 or very close to 0
#
# Important:
# With only one group, EC will always be 0 because there is no other group
# to compare representation against. This test checks coverage behavior,
# not fairness across groups.

perfect_doc_texts = [
    "A woman requested assistance obtaining a passport."
]

perfect_doc_labels = [
    "women"
]

perfect_summary = "A woman requested assistance obtaining a passport."

perfect_ec_result = compute_equal_coverage(
    doc_texts=perfect_doc_texts,
    doc_labels=perfect_doc_labels,
    summary=perfect_summary,
)

print("\n\n=== PERFECT MATCH EC TEST ===")
print("Document text:")
print(perfect_doc_texts[0])

print("\nSummary:")
print(perfect_summary)

print("\nEC(sex):", perfect_ec_result["ec_score"])
print("Overall coverage:", perfect_ec_result["overall_coverage_prob"])
print("Per-group coverage:", perfect_ec_result["per_group_coverage"])
print("Group counts:", perfect_ec_result["group_counts"])


# --------------------------------------------------
# 5. Fair two-group EC sanity check
# --------------------------------------------------
# Goal:
# Check what happens when both groups are represented equally.
#
# Expected outcome:
# - women's coverage should be high
# - men's coverage should be high
# - EC should be low / close to 0

fair_doc_texts = [
    "A woman requested passport assistance.",
    "A man requested employment assistance.",
]

fair_doc_labels = [
    "women",
    "men",
]

fair_summary = """
A woman requested passport assistance.
A man requested employment assistance.
"""

fair_ec_result = compute_equal_coverage(
    doc_texts=fair_doc_texts,
    doc_labels=fair_doc_labels,
    summary=fair_summary,
)

print("\n\n=== FAIR TWO-GROUP EC TEST ===")
print("Document texts:")
for text in fair_doc_texts:
    print("-", text)

print("\nSummary:")
print(fair_summary)

print("\nEC(sex):", fair_ec_result["ec_score"])
print("Overall coverage:", fair_ec_result["overall_coverage_prob"])
print("Per-group coverage:", fair_ec_result["per_group_coverage"])
print("Group counts:", fair_ec_result["group_counts"])


# --------------------------------------------------
# 6. Biased two-group EC sanity check
# --------------------------------------------------
# Goal:
# Check what happens when only one group is represented.
#
# Expected outcome:
# - women's coverage should be high
# - men's coverage should be low
# - EC should be much higher than in the fair two-group test

biased_two_group_doc_texts = [
    "A woman requested passport assistance.",
    "A man requested employment assistance.",
]

biased_two_group_doc_labels = [
    "women",
    "men",
]

biased_two_group_summary = """
A woman requested passport assistance.
"""

biased_two_group_ec_result = compute_equal_coverage(
    doc_texts=biased_two_group_doc_texts,
    doc_labels=biased_two_group_doc_labels,
    summary=biased_two_group_summary,
)

print("\n\n=== BIASED TWO-GROUP EC TEST ===")
print("Document texts:")
for text in biased_two_group_doc_texts:
    print("-", text)

print("\nSummary:")
print(biased_two_group_summary)

print("\nEC(sex):", biased_two_group_ec_result["ec_score"])
print("Overall coverage:", biased_two_group_ec_result["overall_coverage_prob"])
print("Per-group coverage:", biased_two_group_ec_result["per_group_coverage"])
print("Group counts:", biased_two_group_ec_result["group_counts"])

# --------------------------------------------------
# 7. Mismatch sanity check: unrelated summary
# --------------------------------------------------
# Goal:
# Check what happens when the summary is unrelated to the input.
#
# Example:
# - Input says passport assistance
# - Summary says dog assistance
#
# Expected outcome:
# - coverage should be low
# - EC may still be 0 if there is only one group
#
# This shows that EC does NOT measure factual quality by itself.
# It measures whether coverage is balanced across groups.

mismatch_doc_texts = [
    "A woman requested assistance obtaining a passport."
]

mismatch_doc_labels = [
    "women"
]

mismatch_summary = "A woman requested assistance getting a dog."

mismatch_ec_result = compute_equal_coverage(
    doc_texts=mismatch_doc_texts,
    doc_labels=mismatch_doc_labels,
    summary=mismatch_summary,
)

print("\n\n=== MISMATCH EC TEST ===")
print("Document text:")
print(mismatch_doc_texts[0])

print("\nSummary:")
print(mismatch_summary)

print("\nEC(sex):", mismatch_ec_result["ec_score"])
print("Overall coverage:", mismatch_ec_result["overall_coverage_prob"])
print("Per-group coverage:", mismatch_ec_result["per_group_coverage"])
print("Group counts:", mismatch_ec_result["group_counts"])
# ---------------------------------------------------------------------------------
# Equal Coverage (EC) interpretation
#
# EC measures representation fairness across demographic groups.
#
# Lower EC = fairer representation
# Higher EC = more unequal representation
#
# EC compares each subgroup's coverage probability to the OVERALL average
# coverage probability:
#
#     EC(D,S) = (1/K) * Σ | p(overall) - p(group_k) |
#
# where:
# - p(overall): average summary coverage across all feedback
# - p(group_k): coverage for demographic subgroup k
#
# Important:
# EC does NOT directly compare groups to each other.
# Instead, it measures how far each group deviates from the overall average.
#
# Practical interpretation (empirical guideline for this thesis):
#
# EC ≈ 0.00 - 0.05   -> very fair representation
# EC ≈ 0.05 - 0.15   -> mild imbalance
# EC ≈ 0.15 - 0.30   -> noticeable imbalance
# EC > 0.30          -> strong imbalance
#
# Example:
#
# Women coverage = 0.91
# Men coverage   = 0.20
# Overall        = 0.56
#
# EC = (|0.56-0.91| + |0.56-0.20|)/2
#    ≈ 0.36
#
# -> strong representation imbalance
#
# Note:
# EC=1 is generally unrealistic under this formulation because subgroup
# coverage is compared against the moving overall average rather than
# directly against other groups.

# Important:
# Li et al. (2024) do not define formal EC thresholds.
# EC is primarily intended as a relative comparison metric:
#
# Lower EC -> more equal representation
# Higher EC -> less equal representation
#
# Interpretation should focus on differences between systems
# (e.g., baseline vs MBG) rather than absolute values.
#
# Example:
#
# Baseline EC = 0.12
# MBG EC      = 0.05
#
# -> MBG achieves more equal coverage.
# ---------------------------------------------------------------------------------
