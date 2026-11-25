# Counting CVs: job-role classification and class-prevalence estimation on a de-duplicated CV benchmark (2026)

Shah Fahad · independent follow-up study, November 2026

> **This is a new study with new data.** It is not my MSc dissertation and does not reuse its data or results. All numbers below come from `new_study_2026.py` and `results/new_study_results.json` in this repository.

**How this was made:** I directed this study and approved its design. An AI coding assistant (Claude) wrote the code and ran it on my laptop. Every number can be reproduced with one command (see below).

## 1. Why this study
The archived code from my MSc dissertation (2025) compares Naive Bayes, SVM and DistilBERT for résumé classification on the public Kaggle `UpdatedResumeDataSet.csv`. When I reran that code in 2026, I found that only **166 of the 962 résumés in that file are distinct**. With duplicates kept, almost every test résumé has a copy in training and scores look near-perfect. With duplicates removed, SVM reached 61.8% on just 34 unseen résumés, too few to compare models reliably.

This study starts again on a properly sourced, de-duplicated and documented dataset. It answers two questions:
1. **Document level:** how well do Naive Bayes, SVM and DistilBERT tell five software roles apart on CVs they have never seen?
2. **Collection level:** if the classifier is used to estimate *what share* of a batch of CVs belongs to each role, how wrong is simple classify-and-count, and do the adjusted estimators studied by Keith & O'Connor (EMNLP 2018) do better? That paper is binary; this is a five-class setting.

## 2. Data
The source is the **Djinni Recruitment Dataset**, English candidate CVs: anonymised IT job-seeker profiles from 2020–2023, MIT licence (Drushchak & Romanyshyn, UNLP 2024). The file is pinned to one dataset commit and checked by SHA-256. Full details are in [`DATASET_CARD.md`](DATASET_CARD.md).

- **Roles** come from the candidate's self-chosen *Primary Keyword*: Python Developer, Java Developer, Data Analyst, Software Tester (QA or QA Automation) and Website Developer/Designer. The last one needs a JavaScript, PHP or Design keyword plus a web or front-end job title. The models see the CV text only, never the title or keyword.
- **Cleaning:**
  - 78,825 profiles match the five roles;
  - 25,203 were removed as too short (fewer than 50 tokens);
  - **0 exact duplicates** were found;
  - **1 near duplicate** (TF-IDF cosine ≥ 0.90) was found in a screened pool of 9,595.
- **Sample:** 3,000 CVs, 600 per role, split per role into **200 train / 100 validation / 300 test** (1,000 / 500 / 1,500), seed 42. No exact or near duplicate occurs across splits.
- **Synthetic CVs: none.**

## 3. Methods
- **Text cleaning** is unchanged from the archived dissertation code: lower-case, letters only, stop-words removed, lemmatised.
- **Naive Bayes and SVM:** TF-IDF with 5,000 features, `MultinomialNB`, and `SVC(class_weight="balanced")` with RBF kernel and Platt probabilities.
- **DistilBERT:** `distilbert-base-uncased` fine-tuned for 3 epochs, batch 8, max length 256, learning rate 5e-5 (the library default), on CPU (37.5 minutes).
- **No tuning.** All settings are those of the archived dissertation code, fixed before seeing any result. Nothing was tuned on the test set.
- **Prevalence estimators** (each gives the five role shares of a batch):
  - **CC**, classify and count: count the predicted labels.
  - **ACC**, adjusted CC: correct CC using the confusion rates P(predicted | true) measured on the validation split. It solves a constrained least-squares problem on the probability simplex, the multiclass form of Keith & O'Connor's eq. 2.
  - **PCC**, probabilistic CC: average the predicted probabilities.
  - **PACC**: ACC applied to averaged probabilities.
  - **EM**: the Saerens et al. (2002) prior-adjustment algorithm, which Keith & O'Connor note is close to their LR-Implicit method.
- **Batches:** 1,000 batches from the 1,500-CV test pool, 500 of 50 CVs and 500 of 200. Each batch's true role mix is drawn uniformly at random from all possible mixes (Dirichlet(1,…,1)). All models and methods see the same batches.
- **Error measure:** mean absolute error (MAE) of the five estimated shares, in percentage points, averaged over batches.

## 4. Results

### 4.1 Classifying single CVs (1,500 held-out test CVs; 95% bootstrap intervals)
| Model | Accuracy | Macro-F1 |
| --- | --- | --- |
| Naive Bayes | **92.5%** (91.1–93.9) | 0.924 (0.910–0.938) |
| SVM | 90.5% (89.1–92.1) | 0.907 (0.892–0.922) |
| DistilBERT | 91.5% (90.1–92.9) | 0.916 (0.901–0.929) |

- **No model is clearly best.** The intervals overlap, and across five different random splits Naive Bayes averages 91.3% (sd 1.0) and SVM 90.9% (sd 0.8). DistilBERT was run once (seed 42) because of CPU cost.
- **Most errors involve Python Developer,** which has the lowest F1 for every model (0.84–0.89).
  - SVM and DistilBERT often label Java CVs as Python (32 and 17 of 300); DistilBERT does the same with Data Analyst CVs (19).
  - Naive Bayes mostly makes the reverse mistake, labelling Python CVs as Data Analyst, Java or Website.
  - Software Tester is the easiest role (F1 0.93–0.95).
- **The models are not just reading role names.** Removing obvious role words (python, java, javascript, analyst, qa, test, web, frontend, design and similar) leaves Naive Bayes at 89.7% and SVM at 88.1%.

### 4.2 Estimating the role mix of a batch: balanced training (MAE in percentage points; lower is better)
| Model | CC | ACC | PCC | PACC | EM |
| --- | --- | --- | --- | --- | --- |
| Naive Bayes | **1.87** | 1.98 | 6.58 | 2.22 | 4.94 |
| SVM | 2.47 | 1.93 | 2.61 | 1.70 | **1.28** |
| DistilBERT | 2.02 | 1.61 | 2.10 | **1.55** | 1.56 |

When the training mix is balanced and the classifier is about 91% accurate, every estimator is within about 2 points per role, except Naive Bayes with PCC and EM.
- **Naive Bayes** probabilities are under-confident: the average top probability is 0.65 at 92% accuracy, which pulls PCC towards an even split.
- **SVM and DistilBERT:** ACC beats CC in 66% of batches, and the gap grows as the batch mix moves further from the training mix. For DistilBERT, CC rises from 1.57 to 2.37 points from low to high shift while ACC stays at about 1.6.
- **Naive Bayes:** ACC beats CC in only 44% of batches.
- **Batch size:** errors are smaller for batches of 200 than of 50.

### 4.3 Estimating the role mix of a batch: skewed training (NB and SVM)
Following Keith & O'Connor's shifted setting, each model was retrained with one role making up two-thirds of the training data (200 CVs, the other four roles 25 each). The run was repeated with each of the five roles as the dominant one and averaged. The test batches are the same as in 4.2.

| Model | CC | ACC | PCC | PACC | EM |
| --- | --- | --- | --- | --- | --- |
| Naive Bayes | 31.9 | 13.7 | 28.3 | **9.4** | 32.0 |
| SVM | 17.6 | 5.9 | 5.4 | **2.3** | 3.5 |

This is the case where classify-and-count breaks down:
- With a skewed training mix, CC misestimates role shares by 18–32 points on average.
- Adjusting with validation-set confusion rates cuts that to 6–14 points (ACC) and 2–9 points (PACC).
- With Naive Bayes, EM fails. In one example (Data Analyst dominant), NB labelled 99% of test CVs as the dominant role, which leaves almost no signal for any correction method.

This matches Keith & O'Connor's binary Yelp finding that CC fails under prior shift while adjusted methods hold up, here in a five-class setting.

## 5. Limitations
- **Labels** are self-reported Primary Keywords from one platform (the Ukrainian IT market, 2020–2023). Djinni profiles are shorter and more uniform than full PDF résumés, so accuracy here will not transfer directly to other CV collections.
- **Single runs:**
  - DistilBERT was run once with default settings.
  - The skewed-training experiment covers NB and SVM only.
  - No hyperparameters were tuned, for any model.
- **Batches and confusion rates:**
  - All batches come from the same 1,500 test CVs, so they are not independent samples.
  - ACC's confusion rates come from only 100 validation CVs per role.
- **Not studied:** I did not study **uncertainty intervals or their coverage**, the main open problem in Keith & O'Connor. The batch mixes are simulated, not real application streams.
- **Duplicate screening** covered a 9,595-CV pool. Near duplicates outside that pool could not affect the sample, but the same person may hold differently worded profiles that cannot be detected in anonymised data.

**Natural next steps:** interval estimates and their coverage for multiclass prevalence; DistilBERT and implicit-likelihood estimators under skewed training; real time-ordered application batches.

## 6. Reproduce everything (one command)
```
pip install -r requirements.txt
python new_study_2026.py
```
The script downloads the data (237 MB, SHA-256 checked), builds the sample, trains all three models (DistilBERT about 35–45 minutes on CPU) and writes `results/new_study_results.json`. `python new_study_2026.py --quick` skips DistilBERT training and reuses the saved DistilBERT outputs in `results/distilbert_outputs_seed42.npz`; it reproduces every number above exactly.

## References
- Drushchak, N. & Romanyshyn, M. (2024). Introducing the Djinni Recruitment Dataset: A Corpus of Anonymized CVs and Job Postings. *UNLP @ LREC-COLING 2024*, 8–13.
- Keith, K. A. & O'Connor, B. (2018). Uncertainty-aware generative models for inferring document class prevalence. *EMNLP 2018*, 4575–4585.
- Saerens, M., Latinne, P. & Decaestecker, C. (2002). Adjusting the outputs of a classifier to new a priori probabilities: a simple procedure. *Neural Computation* 14(1), 21–41.
- Forman, G. (2008). Quantifying counts and costs via classification. *Data Mining and Knowledge Discovery* 17(2), 164–206.
