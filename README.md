# ASL Alphabet - Sign Language Text Writer

A real-time American Sign Language (ASL) alphabet detector that lets you write text using hand signs in front of your webcam.

---

## How it works

1. **Data cleaning** - scans the ASL dataset, filters bad images, and extracts 21 hand-landmark vectors using MediaPipe.
2. **Training** - trains an SVM classifier on the extracted landmarks.
3. **Web app** - open the web UI; hold a sign steady for ~1 second to add the letter to the text area.

---

## Requirements

- Python 3.10 to 3.13
- A webcam

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset

Download the **ASL Alphabet** dataset from Kaggle:
https://www.kaggle.com/datasets/grassknoted/asl-alphabet

After downloading, extract it so the folder structure looks like this:

```
vpo/
data/
    asl_alphabet_train/
        asl_alphabet_train/
            A/
            B/
            ...
            del/
            nothing/
            space/
```

---

## Run

### Step 1 - Clean the dataset and extract landmarks

```bash
python data_cleaning.py
```

Produces: X.npy, y.npy, classes.npy

### Step 2 - Train the model

```bash
python train_model_svm.py
```

Produces: model_svm.pkl, confusion_matrix_svm.png

### Step 3 - Launch the app

```bash
python app.py
```

---

## App controls

| Action | How |
|--------|-----|
| Write a letter | Hold the sign steady for ~1 second |
| Space | Hold the space sign, or press Space on keyboard |
| Delete last character | Hold the del sign, or press Delete / Backspace on keyboard |
| Clear all text | Click the CLEAR button |
| Copy text | Click the COPY button |

To write the same letter twice (e.g. AA), hold the sign for a second, release briefly, then hold it again.

---

## Project files

| File | Description |
|------|-------------|
| hand_detector.py | MediaPipe hand landmark extractor (shared module) |
| data_cleaning.py | Dataset cleaning and landmark extraction |
| train_model_svm.py | SVM training and evaluation |
| app.py | FastAPI web app |
| vpo.ipynb | Notebook with MediaPipe exploration |
