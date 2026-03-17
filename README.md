# DocFusion — Rihal CodeStacker 2026 ML Challenge
### by Moza Amur

---

## My Journey With This Challenge

When I first read the problem statement, my immediate reaction was excitement mixed 
with a healthy amount of uncertainty. 

This README is not just documentation. It is an honest account of how I thought 
through this problem, what failed, what worked, and what I would do differently.

---

## First Thoughts

The first question I asked myself was: *where do I even start?*

Three datasets, four levels, an autograder harness, and a web UI 
all within a tight deadline. I decided to treat this like a real engineering problem rather than 
a homework assignment. That meant:

1. Understand the data before writing a single line of model code
2. Build a working pipeline end to end, even if imperfect
3. Improve incrementally rather than trying to be perfect from the start

---

## Dataset Discovery  (Harder Than Expected)

This was honestly the hardest part for me.

The three datasets — SROIE, CORD, and Find-It-Again — are very different from each 
other in structure, format, and purpose. I spent significant time just figuring out:

- What fields each dataset actually contains
- How the label files are formatted
- Why Find-It-Again's `train.txt` is a CSV with complex JSON inside some columns,
  not a simple two-column file like I assumed

For example, my first attempt to parse Find-It-Again labels failed immediately:
```
ValueError: invalid literal for int() with base 10: 'annotation,handwritten'
```
That error taught me something important — always inspect the raw file before 
assuming its format. After printing the first few lines I realized it was a CSV 
with a header row and multiple columns. A small discovery, but it cost me time.

I also discovered through a GitHub issue on the challenge repo that CORD does not 
contain `vendor` or `date` fields in its ground truth — the organizers confirmed 
it is acceptable to return `null` for fields that do not exist in a dataset. That 
clarification changed how I thought about the pipeline's job: not perfect extraction, 
but robust handling of heterogeneous inputs.

---

## My Approach

### OCR Engine Selection

I evaluated three options:

| Option | Accuracy | Speed | Memory | Windows Support |
|---|---|---|---|---|
| Tesseract | Medium | Medium | Low | Difficult to install |
| TrOCR | High | Slow | Very High | Complex |
| PaddleOCR | Good | Fast | Low | Easy |

I chose **PaddleOCR** because Level 4 explicitly measures inference speed and memory 
usage. A highly accurate but slow model would hurt my score where it matters most.

### Field Extraction (Rule-Based First)

My first instinct was to use a language model for extraction. But I stepped back and 
asked: do I really need that? The fields are structured and receipts follow patterns. 
Regex rules with PaddleOCR output should work.

My first results on 626 SROIE receipts:
```
Vendor accuracy : 41.5%
Date accuracy   : 45.8%
Total accuracy  : 46.3%
```

Not great. I spent time iterating on the rules — improving vendor detection by 
looking for company indicators like `SDN BHD`, `ENTERPRISE`, `LTD`, fixing date 
patterns to handle merged OCR outputs like `12-01-1921` instead of `12-01-19`, 
and making the total extractor prefer lines near the bottom of the receipt.

After several iterations the scores did not improve dramatically. This taught me 
something I will remember: **rule-based systems have a ceiling**. The OCR noise is 
sometimes too unpredictable for regex to handle reliably. With more time I would 
replace the regex extractor with a fine-tuned named entity recognition model or a 
small language model specifically trained on receipt text.

Final extraction scores:
```
Vendor accuracy : 41.7%
Date accuracy   : 45.8%
Total accuracy  : 44.2%
```

Honest assessment: these numbers are modest. The pipeline is functional and handles 
diverse layouts, but accuracy needs improvement.

### Anomaly Detection (The ML Part)

This is where the real machine learning happens. My approach:

1. Convert each receipt into a 15-dimensional feature vector
2. Train a Random Forest classifier on Find-It-Again labels

**Text-based features I used:**
- Is vendor missing?
- Is date missing?
- Is total missing?
- Total amount value and its log
- Number of OCR lines
- Average line length
- Number of price patterns in text
- How many times the total repeats

**Image-based features I added:**
- Laplacian variance (measures image sharpness/noise)
- Mean brightness and standard deviation
- Edge density via Canny edge detection
- Block variance difference (detects inconsistent regions)

My first honest evaluation on the validation set:
```
Genuine:  84% precision, 91% recall
Forged:   25% precision, 15% recall
Overall accuracy: 78%
F1 cross-validation: 0.05
```

The model was basically ignoring forged receipts entirely. I realized two things:

First, the dataset is heavily imbalanced (only 16% forged). A model that always 
says "genuine" gets 84% accuracy without learning anything. I fixed this by 
upsampling the forged class during training.

Second, and more importantly — the forgeries in Find-It-Again are **visual**. 
A tampered pixel or copy-pasted region does not change the text fields much. 
My text features alone were never going to catch visual forgery reliably. 
This is why I added the image-based features.

**Why Random Forest over other models?**

I considered neural networks but rejected them for this challenge specifically 
because Level 4 penalizes slow inference and high memory usage. Random Forest 
gives a good balance of accuracy, speed, and small model size. With more time 
and data I would explore a Vision Transformer or EfficientNet for the visual 
forgery detection component.

---

## What I Would Improve With More Time

1. **Better extraction** — Replace regex with a fine-tuned NER model or 
   LayoutLM, which understands both text and spatial layout of documents

2. **Better forgery detection** — Use a CNN or Vision Transformer to detect 
   visual tampering at the pixel level rather than relying on high-level features

3. **CORD integration** — I used CORD for diversity but did not fully leverage 
   its item-level annotations for richer feature engineering

4. **Proper cross-dataset evaluation** — Test the pipeline on CORD and 
   Find-It-Again together to measure true generalization

5. **Docker containerization** — Did not have time to complete the Dockerfile

---

## Honest Acknowledgments

The Streamlit web UI (`app.py`) was built with AI assistance. I have solid Python 
and ML experience but limited frontend and UI experience. I used AI to help 
structure the Streamlit layout while I focused my energy on the core pipeline — 
the OCR, extraction logic, feature engineering, and model training. I believe in 
being transparent about this.

The core ML pipeline, feature engineering decisions, model selection reasoning, 
and all the debugging were done by me working through the problem step by step.

---

## What I Enjoyed

Trying different solutions to improve performance was genuinely enjoyable. Every 
time a metric improved — even slightly — it felt like progress. I especially 
enjoyed the feature engineering process for anomaly detection: thinking about 
what signals a forged receipt might leave behind, and translating those intuitions 
into numerical features. That bridge between human reasoning and machine learning 
is what I find most interesting about this field.

---

## Project Structure
```
docfusion-codestacker-2026/
├── solution.py          # Harness interface (Level 4)
├── app.py               # Streamlit web UI (Level 3)
├── check_submission.py  # Local validation script
├── requirements.txt     # Dependencies
├── README.md
├── src/
│   ├── extractor.py     # OCR + field extraction (Level 2)
│   ├── anomaly.py       # Anomaly detection ML model (Level 3)
│   └── evaluate.py      # Evaluation script
├── notebooks/
│   └── 01_eda.ipynb     # EDA notebook (Level 1)
├── dummy_data/          # Local test data from challenge repo
│   ├── train/
│   └── test/
└── models/
    └── anomaly/         # Saved model artifacts
```

## Setup
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Run Web UI
```bash
streamlit run app.py
```

## Run Local Validation
```bash
python check_submission.py --submission .
```

## Train Anomaly Model
```bash
python src/anomaly.py
```

## Tech Stack

| Tool | Purpose |
|---|---|
| PaddleOCR | OCR engine for reading receipt text |
| scikit-learn | Random Forest anomaly detection |
| OpenCV | Image feature extraction |
| Streamlit | Web UI dashboard |
| Pandas | Data exploration and EDA |
| Jupyter | EDA notebook |

## Datasets

| Dataset | Source | Role |
|---|---|---|
| SROIE | Kaggle | Primary extraction training data |
| CORD | HuggingFace | Layout diversity and robustness |
| Find-It-Again | L3i / Kaggle | Forgery detection labels |
