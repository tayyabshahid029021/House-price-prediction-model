# 🏠 House Price Predictor — King County, USA

A machine learning project that predicts house prices based on location, size, condition, and other key features using real estate sales data from King County, USA.

---

## 📊 Models & Results

| Model | R² Score |
|-------|----------|
| Linear Regression | 0.7336 |
| Gradient Boosting | **0.9215** |

Gradient Boosting significantly outperformed Linear Regression, capturing non-linear relationships between features and price much more effectively.

---

## 📁 Project Structure

```
house-price-predictor/
│
├── housesales.ipynb      # Main notebook (EDA + Model Training)
├── app.py                # Prediction script
├── kc_house_data.csv     # Dataset
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

---

## 🔍 What's Inside the Notebook

- **Exploratory Data Analysis** — Visualizations exploring relationships between price, square footage, location, bedrooms, floors, and waterfront properties
- **Feature Engineering** — Data cleaning, date conversion, feature selection
- **PCA Analysis** — Dimensionality reduction, identifying the number of components needed to explain 95% of the variance
- **Linear Regression** — Baseline model (R² = 0.7336)
- **Gradient Boosting** — High-accuracy model (R² = 0.9215)

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

---

## ▶️ Run Locally

```bash
git clone https://github.com/tayyabshahid029021/House-Price-Prediction-Model.git
cd House-Price-Prediction-Model
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
python app.py
```

Or explore the full analysis and model training process:
```bash
jupyter notebook housesales.ipynb
```

---

## 📦 Dataset

King County House Sales Dataset — 21,000+ records
Source: [Kaggle](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)

---

## 👤 Author

**Muhammad Tayyab Nawaz**
[GitHub](https://github.com/tayyabshahid029021)
