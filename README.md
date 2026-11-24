# cv-role-prevalence-2026

**Job-role classification and class-prevalence estimation on a de-duplicated, licence-documented CV benchmark.** An independent 2026 follow-up to my MSc dissertation on résumé classification, using new data. Shah Fahad.

- **Data:** 3,000 anonymised IT CVs, 600 for each of five roles (Python Developer, Java Developer, Data Analyst, Software Tester, Website Developer/Designer), taken from the public [Djinni Recruitment Dataset](https://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-english) (MIT licence). Exact and near duplicates were screened out before splitting. See [`DATASET_CARD.md`](DATASET_CARD.md).
- **Classification:** Naive Bayes 92.5%, SVM 90.5%, DistilBERT 91.5% accuracy on 1,500 held-out CVs. The 95% intervals overlap, so no model is clearly best.
- **Prevalence estimation** (after Keith & O'Connor, EMNLP 2018, extended to five classes):
  - With balanced training, all estimators are within about 2 percentage points per role.
  - When the training mix is skewed, classify-and-count misestimates role shares by 18–32 points.
  - Adjusted classify-and-count cuts that to 2–14 points.

Full write-up, tables and limitations: [`NEW_STUDY_2026.md`](NEW_STUDY_2026.md). Raw numbers: [`results/new_study_results.json`](results/new_study_results.json).

## Reproduce
```
pip install -r requirements.txt
python new_study_2026.py            # everything, including DistilBERT (~35-45 min on CPU)
python new_study_2026.py --quick    # skips DistilBERT training, reuses its saved outputs
```
The CV texts are not stored in this repository. The script downloads them from Hugging Face (237 MB, SHA-256 checked) and rebuilds the exact sample.

Code: MIT licence ([`LICENSE`](LICENSE)). Data: licensed separately by its publishers; please cite Drushchak & Romanyshyn (2024) if you use it.
