# Dataset card: Five-role CV benchmark (2026 follow-up study)

**This is new data for a 2026 follow-up study. It is not dissertation data**, and it is unrelated to the public Kaggle `UpdatedResumeDataSet.csv` used by the archived MSc dissertation code.

## Source (the only source used)
| Item | Value |
| --- | --- |
| Name | Djinni Recruitment Dataset, English candidate CVs (`lang-uk/recruitment-dataset-candidate-profiles-english`) |
| Publisher | lang-uk / Stereotypes-in-LLMs project, with data provided by the Djinni IT job platform (Ukraine) |
| Paper | Drushchak, N. & Romanyshyn, M. (2024). *Introducing the Djinni Recruitment Dataset: A Corpus of Anonymized CVs and Job Postings.* UNLP Workshop @ LREC-COLING 2024, pp. 8–13. https://aclanthology.org/2024.unlp-1.2 |
| Dataset URL | https://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-english |
| Exact file | `data/train-00000-of-00001.parquet` at dataset commit `86255174c6c378a2b52cd7e81d79002c64d899a6` |
| SHA-256 | `a7b1378e0b231507c377659bf2c80fa4ba60c4e31e1107b13425d103879ecc3b` (checked by the script) |
| Size | 237,415,736 bytes; 210,250 anonymised English CV profiles posted 2020–2023 |
| Licence | **MIT** (Hugging Face dataset card). The companion code repository https://github.com/Stereotypes-in-LLMs/recruitment-dataset is Apache-2.0. The card lists research uses (classification, fairness benchmarking, recommender systems) and asks users to cite the paper above. |
| Downloaded | 24 November 2026 |
| Local copy | `data_raw/djinni/candidates_en.parquet`, downloaded by the script; **not redistributed in this repository** |

**Synthetic résumés: none.** No synthetic, scraped or private CVs were added.

**Considered but not used:** ResumeAtlas (`ahmedheakl/resume-atlas`, MIT on Hugging Face). Not used because its CVs are compiled from earlier web-scraped collections whose original terms are unclear, and it appears to include the Kaggle dataset used by the archived dissertation code.

## How the five roles were defined
Each Djinni profile has a **Primary Keyword** chosen by the candidate and a free-text **Position** title. Roles (the five targeted in the dissertation brief):

| Role | Rule |
| --- | --- |
| Python Developer | Primary Keyword = `Python` |
| Java Developer | Primary Keyword = `Java` |
| Data Analyst | Primary Keyword = `Data Analyst` |
| Software Tester | Primary Keyword = `QA` or `QA Automation` (final sample: 471 QA, 129 QA Automation) |
| Website Developer/Designer | Primary Keyword = `JavaScript`, `PHP` or `Design`, **and** Position title contains web / front-end / frontend / HTML / WordPress / Webflow (final sample: 545 JavaScript, 34 Design, 21 PHP, so this class is mostly front-end developers) |

The text classified is the `CV` field only (highlights + "more info" text). The Position title and Primary Keyword are **not** given to the models.

## Cleaning and de-duplication (all in `new_study_2026.py`)
| Step | Count |
| --- | --- |
| Profiles in source | 210,250 |
| Profiles matching the five roles | 78,825 (Data Analyst 2,271; Java 11,463; Python 7,588; Tester 30,845; Website 26,658) |
| Removed: fewer than 50 tokens after cleaning | 25,203 |
| Removed: **exact duplicates** (same cleaned text, whole five-role pool) | **0** |
| Screened pool: up to 2,000 per role, randomly drawn (seed 42) | 9,595 (Data Analyst has only 1,595 long enough) |
| Removed: **near duplicates** (TF-IDF cosine ≥ 0.90 to an earlier kept CV, across all roles) | **1** (Java). Sensitivity: 3 CVs at ≥ 0.80, 0 at ≥ 0.95 |
| Final sample, drawn at random (seed 42) | **3,000 CVs, 600 per role** |

Text cleaning is the archived dissertation code's `clean_text` (lower-case, letters only, NLTK stop-words removed, WordNet lemmatisation, words of 3+ letters). Median length is 88–97 cleaned tokens per role.

Splits, per role and stratified (seed 42): **train 200 / validation 100 / test 300** (1,000 / 500 / 1,500 in total). Because the whole sample was de-duplicated first, no exact or near duplicate occurs across splits; the script also asserts that no cleaned text appears in two splits.

## Known limitations
- **Labels are self-reported** Primary Keywords, not expert annotations, and roles overlap (e.g. QA Automation engineers who write Java or Python).
- **One platform, one region:** Ukrainian IT market, 2020–2023, English-language profiles. Djinni profiles are shorter and more uniform than full PDF résumés.
- The upstream release appears to be already de-duplicated (0 exact duplicates). Near-duplicate screening covered the 9,595-CV screened pool, not all 78,825 profiles. The same person may hold several differently worded profiles; that cannot be detected from anonymised data.
- "Website Developer/Designer" is dominated by front-end JavaScript developers; very few are pure designers.

## Privacy
The profiles are anonymised by the publisher. The processed sample (`data/five_roles_2026.parquet`) is rebuilt locally by the script and is excluded from this repository (`.gitignore`); it should not be re-uploaded or quoted. Report only aggregate results.
