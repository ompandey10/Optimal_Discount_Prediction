# Optimal Discount Prediction for Retail Products

This repository contains the structured code and outcomes for Case Study 160: Optimal Discount Prediction for Retail Products using Machine Learning.

## 📂 Project Structure

```text
Optimal_Discount_Prediction_Case_Study/
├── data/
│   └── retail_data.csv          # Generated dataset
├── models/
│   ├── best_model.pkl           # Saved XGBoost model
│   └── model_results.json       # Algorithm comparison metrics
├── src/
│   ├── data_generator.py        # Script to generate synthetic data
│   └── model_trainer.py         # Script to train and evaluate ML models
├── app/
│   └── app.py                   # Streamlit application for deployment
├── reports/
│   └── Case_Study_Report.md     # Final analysis and answers to objectives
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation (this file)
```

## 🚀 How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate the Dataset:**
   ```bash
   python src/data_generator.py
   ```

3. **Train the Models:**
   ```bash
   python src/model_trainer.py
   ```

4. **Launch the Streamlit Web Application:**
   ```bash
   streamlit run app/app.py
   ```

## 📊 Summary of Findings
- The relationship between discount and sales is non-linear (diminishing returns).
- XGBoost performed the best in predicting sales with an R2 score of ~0.95.
- An optimal discount can be recommended by simulating sales and calculating expected revenue across different discount brackets.
