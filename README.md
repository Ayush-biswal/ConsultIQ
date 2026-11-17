# ConsultIQ — Mini Project

A simple Streamlit application for analysing e-consultation feedback.

## Features
- Create consultations and sections
- Submit individual comments
- Import feedback from CSV
- Classify comments as Positive, Negative or Neutral
- Extract frequent words and bigrams
- Flag potentially important comments using simple rules
- View dashboard analytics

## Setup (Windows)

1. Install Python 3.10 or newer.
2. Extract this folder.
3. Open the folder in VS Code.
4. Open Terminal > New Terminal.
5. Create a virtual environment:

   ```powershell
   py -m venv .venv
   .venv\Scripts\activate
   ```

6. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

7. Train the starter model:

   ```powershell
   python ml/train_model.py
   ```

8. Start the application:

   ```powershell
   streamlit run app.py
   ```

The app opens in your browser, usually at http://localhost:8501.

## CSV format

The CSV file must contain:
- `section`: section name (must already exist under the selected consultation)
- `comment`: feedback text

Example:

```csv
section,comment
Road Infrastructure,The road has many potholes and needs repair.
Public Transport,Buses are clean and arrive on time.
Pedestrian Safety,There is a zebra crossing near the school.
```

## Important note about the model

`data/training_data.csv` is a small starter dataset so the project runs immediately. It is for demonstration and development, not a strong research-quality model. For the final academic evaluation, replace or augment it with a larger labelled three-class dataset such as TweetEval, then evaluate on a separate civic-feedback test set. Do not report training accuracy as test accuracy.

## Database

The SQLite database is created automatically at `database/consultiq.db` when the app starts. To reset all application data, stop Streamlit and delete that file.

## Project structure

```text
ConsultIQ/
├── app.py
├── requirements.txt
├── README.md
├── database/
│   └── db.py
├── modules/
│   ├── preprocessing.py
│   ├── sentiment.py
│   ├── keywords.py
│   └── priority.py
├── ml/
│   └── train_model.py
└── data/
    ├── training_data.csv
    └── sample_feedback.csv
```
