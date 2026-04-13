# DocFusion Rihal CodeStacker 2026 ML Challenge
by Moza Amur

---

## What is This Project?

DocFusion is an intelligent document processing pipeline. You give it a receipt image, it reads the text, pulls out the important fields like vendor name, date, and total amount, and then decides if the receipt looks genuine or forged.

I built this for the Rihal CodeStacker 2026 ML Challenge.
This README is my honest story of how I built it — what I tried, what failed, what worked, and what I would do differently.

---

## The Problem

Thousands of receipts and invoices are processed every day in organizations. Most of the information is locked inside scanned images. Someone has to read them manually, type the data into a system, and check if anything looks suspicious. That is slow, expensive, and error-prone.

DocFusion automates this. Upload an image, get structured data back.

---

## Where I Started

When I first read the challenge, I asked myself: where do I even start?

Three datasets, four levels, an autograder, and a web UI (all within a tight deadline). I decided to treat it like a real engineering problem. That meant understanding the data first before writing any model code, building something that works end to end even if imperfect, and improving step by step.

---

## The Datasets

I used three datasets. They each have a different job.

**SROIE** — 973 scanned English receipts with labeled fields (vendor, date, total). I used this to measure how accurate my field extraction is.

**CORD** — 1000 receipts with diverse layouts from different shops and countries. I used this to test that the pipeline does not break on unusual formats.

**Find-It-Again** — 988 receipts where 163 of them are forged. Realistic forgeries — copy-paste attacks, tampered text. I used this to train the anomaly detection model.

One thing I discovered early: these three datasets cannot be used interchangeably. CORD does not have vendor or date in its ground truth. SROIE has no forgery labels. Only Find-It-Again has forgery labels. Each dataset has its own purpose and I had to respect that.

I also found out through a GitHub issue on the challenge repo that returning null for missing fields is acceptable. That changed my thinking — the goal is not perfect extraction, it is a robust pipeline that handles anything you throw at it.

---

## Dataset Sizes

| Dataset | Receipts | Forged | Size on Disk |
|---|---|---|---|
| SROIE | 973 | None | ~200MB |
| CORD | 1,000 | None | ~1.8GB |
| Find-It-Again | 988 | 163 (16%) | ~150MB |
| Total | 2,961 | 163 | ~2.15GB |

---

## Choosing the OCR Engine

I looked at three options before starting.

**Tesseract** is the classic open source OCR. It works but is not very accurate on noisy scanned receipts, and it is hard to install on Windows.

**TrOCR** is a modern transformer-based model with high accuracy. The problem is it is very heavy on memory and slow on CPU. The challenge measures inference speed and memory in Level 4, so this would hurt my score where it matters.

**PaddleOCR** is fast, accurate on receipts, works well on Windows, and is lightweight enough for production. This was the clear choice.

For the cloud demo I used **Tesseract via pytesseract** because PaddleOCR cannot install on the Streamlit Cloud server. The accuracy is lower but it is good enough for a demo. The full pipeline with PaddleOCR runs locally.

I also tried EasyOCR as a middle ground for the cloud, it is more accurate than Tesseract and simpler than PaddleOCR. But it requires PyTorch which caused DLL errors on Windows and installation failures on the cloud. So I dropped it.

---

## Field Extraction

My first instinct was to use a language model to extract fields. Then I stepped back and asked: do I really need that? Receipts follow patterns. Regex should work.

I built a rule-based extractor using PaddleOCR output:
- Vendor: look for company indicators like SDN BHD, ENTERPRISE, LTD in the first few lines
- Date: match common date patterns like DD/MM/YYYY or YYYY-MM-DD
- Total: find lines with total keywords near the bottom of the receipt

My first accuracy on 626 SROIE receipts:
```
Vendor : 41.5%
Date   : 45.8%
Total  : 46.3%
```

I spent time improving the rules, better vendor detection, fixing date patterns, smarter total extraction. After several iterations the scores barely moved. This taught me something important: rule-based systems have a ceiling. OCR noise is too unpredictable for regex to handle reliably.

Final scores after all improvements:
```
Vendor : 41.7%
Date   : 45.8%
Total  : 44.2%
```

Honest assessment: these numbers are modest. The pipeline works and handles diverse layouts, but with more time I would replace the regex extractor with LayoutLM or a fine-tuned NER model that understands both text content and spatial position on the document.

---

## Anomaly Detection — The ML Part

This is where the actual machine learning happens. The goal is to predict is_forged: 0 for genuine, 1 for forged.

I went through four iterations.

**Iteration 1: Text features only**

I converted each receipt into 10 numbers:
- Is vendor missing?
- Is date missing?
- Is total missing?
- Total amount value and its log
- Number of OCR lines
- Average line length
- Number of price patterns in the text
- How many times total repeats
- Number of lines with numbers

Results on validation set:
```
Genuine: 84% precision, 91% recall
Forged:  25% precision, 15% recall
Overall: 78% accuracy
Forged F1: 0.19
```

The model was basically ignoring forged receipts. Two problems. First, the dataset is heavily imbalanced — only 16% forged. A model that always says genuine gets 84% accuracy without learning anything. Second, the forgeries in Find-It-Again are visual — tampered pixels and copy-paste. Text features alone cannot catch that.

**Iteration 2: Text + basic OpenCV image features**

I added brightness statistics, edge density, and block variance to detect inconsistent regions. The forged recall actually got worse — more features confused the model because we had too little training data.
```
Forged F1: 0.14
```

**Iteration 3: Text + LBP + FFT + compression artifacts**

I added Local Binary Patterns for texture analysis and FFT frequency analysis. These are classical computer vision techniques that can detect copy-paste patterns. Still not much improvement.
```
Forged F1: 0.10
```

**Iteration 4: Switch to XGBoost**

I switched from Random Forest to XGBoost with scale_pos_weight to handle the class imbalance automatically. XGBoost builds trees sequentially — each tree learns from the mistakes of the previous one. Combined with the image features, this gave the best result.
```
Genuine: 84% precision, 90% recall
Forged:  27% precision, 18% recall
Overall: 78% accuracy
Forged F1: 0.22
Cross-validation F1: 0.16
```

**Full comparison:**

| Iteration | Features | Forged F1 |
|---|---|---|
| 1 | Text only, Random Forest | 0.19 |
| 2 | Text + OpenCV basics, Random Forest | 0.14 |
| 3 | Text + LBP + FFT, Random Forest | 0.10 |
| 4 | Text + image features, XGBoost | 0.22 |

XGBoost won.

---

## Why XGBoost Over Random Forest?

I have used XGBoost a lot in previous projects. It is powerful and often gives better accuracy than Random Forest on tabular data because it builds trees sequentially and corrects its own mistakes.

My initial plan was actually to use Random Forest because I was worried about Level 4 performance constraints (XGBoost models can be larger and slower). But after Random Forest kept performing poorly on forged detection, I switched.

The key feature that made the difference is scale_pos_weight. This tells XGBoost that forged receipts are rare and should be weighted more heavily during training. It is built specifically for imbalanced datasets, which is exactly our problem.

With more time I would run a proper comparison — train both, measure inference time and memory on the same machine, and make a data-driven decision. But given the results, XGBoost was clearly the right call here.

---

## What I Would Improve

Better extraction: Replace regex with LayoutLM or a fine-tuned NER model. LayoutLM understands both text and where it appears on the page, which is exactly what receipt extraction needs.

Better forgery detection: The core weakness is that our features cannot see pixel-level tampering. A CNN or Vision Transformer trained to spot copy-paste artifacts would catch what our features miss. I tried to use EfficientNet via PyTorch but hit DLL errors on Windows that I could not resolve.

More training data: 94 forged training samples is very little. The model cannot learn reliable patterns from that. With more forged examples the XGBoost model would improve significantly.

Docker containerization: Did not have time to complete the Dockerfile.

Full CORD integration: I downloaded CORD and confirmed the pipeline handles it without errors, but did not leverage its item-level annotations for richer feature engineering.

---

## Live Demo

The cloud demo uses Tesseract OCR instead of PaddleOCR. Extraction accuracy is lower but the interface is fully functional.

Live app: https://docfusion-codestacker-2026-guwbvmje4cv2srqpd5tddw.streamlit.app

To run the full version locally with PaddleOCR:
```
git clone https://github.com/mozaalismaili/docfusion-codestacker-2026.git
cd docfusion-codestacker-2026
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## Honest Acknowledgments

The Streamlit web UI was built with AI assistance. I have solid Python and ML experience but limited UI experience. I used AI to help structure the layout while I focused on the core pipeline — OCR, extraction, feature engineering, and model training. I think being honest about this matters.

Everything else — the dataset exploration, feature engineering decisions, model selection, debugging, and the iterations that did not work — was done by me working through the problem step by step.

---

## What I Enjoyed

Trying different solutions to improve performance was genuinely the most enjoyable part. Every small improvement felt like progress. I especially liked the feature engineering process — thinking about what a forged receipt might look like and translating that intuition into numbers the model can learn from. That connection between human reasoning and machine learning is what I find most interesting about this field.

---

## Project Structure
```
docfusion-codestacker-2026/
├── solution.py           -- harness interface (Level 4)
├── app.py                -- full Streamlit app using PaddleOCR (local)
├── app_cloud.py          -- cloud Streamlit app using Tesseract
├── check_submission.py   -- local validation script
├── requirements.txt      -- dependencies
├── packages.txt          -- system dependencies for cloud
├── README.md
├── src/
│   ├── extractor.py      -- PaddleOCR + regex extraction (Level 2)
│   ├── anomaly.py        -- XGBoost anomaly detection (Level 3)
│   └── evaluate.py       -- accuracy measurement
├── notebooks/
│   └── 01_eda.ipynb      -- EDA notebook (Level 1)
├── dummy_data/           -- test data from challenge repo
│   ├── train/
│   └── test/
└── models/
    └── anomaly/          -- saved model files
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| PaddleOCR | OCR engine for reading receipt text (local) |
| Tesseract | OCR engine for cloud deployment |
| XGBoost | Anomaly detection model |
| scikit-learn | Feature scaling and evaluation |
| OpenCV + scikit-image | Image feature extraction |
| Streamlit | Web UI |
| Pandas | Data exploration |
| Jupyter | EDA notebook |

---

## Datasets

| Dataset | Source | Role |
|---|---|---|
| SROIE | Kaggle | Extraction evaluation |
| CORD | HuggingFace | Pipeline robustness testing |
| Find-It-Again | L3i / Kaggle | Forgery detection training |

---

## About Me

Moza Amur
Final-year Computer Science student
Sultan Qaboos University
Specialization: Artificial Intelligence and Data Science
