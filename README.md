# 🏠 House Price Predictor — King County, USA

A machine learning web app that predicts house prices based on location, 
size, condition, and other key features using real estate data from King County, USA.

🌐 **Live App:** [Click here to try it](https://hamza-house-predictor.streamlit.app/)

---

## 📊 Models & Results

| Model | R² Score |
|-------|----------|
| Linear Regression | ~0.65 |
| Gradient Boosting | **0.88** |

Gradient Boosting outperformed Linear Regression significantly, 
capturing non-linear relationships between features and price.

---

## 📁 Project Structure

house-price-predictor/
│
├── housesales.ipynb      # Main notebook (EDA + Model Training)
├── app.py                # Streamlit web app
├── kc_house_data.csv     # Dataset
├── requirements.txt      # Dependencies
└── README.md             # Project documentation

---

## 🔍 What's Inside the Notebook

- **Exploratory Data Analysis** — 10+ visualizations (price vs sqft, location, bedrooms, waterfront)
- **Feature Engineering** — Date conversion, feature selection
- **Linear Regression** — Baseline model
- **Gradient Boosting** — High accuracy model (R² = 0.88)
- **PCA Analysis** — 19 features → 14 components cover 95% variance

---

## 🌐 Streamlit Web App Features

- Interactive sliders for all house features
- Real-time price prediction
- Low & High price range estimate
- Model performance displayed

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit

---

## ▶️ Run Locally

```bash
git clone https://github.com/hamzaisrar0001/house-price-predictor.git
cd house-price-predictor
pip install -r requirements.txt
streamlit run app.py
```

---

## 📦 Dataset

King County House Sales Dataset — 21,000+ records  
Source: [Kaggle](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)

---

## 👤 Author

**Muhammad Hamza Israr**  
[GitHub](https://github.com/hamzaisrar0001) · [LinkedIn](https://linkedin.com/in/hamzaisrar0001)
