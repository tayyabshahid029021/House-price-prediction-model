import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pickle
import os

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ─── Model train karo (sirf ek baar) ───
@st.cache_resource
def load_model():
    data = pd.read_csv("kc_house_data.csv")
    data['date'] = pd.to_datetime(data['date'])
    data['date'] = (data['date'].dt.year == 2014).astype(int)
    labels = data['price']
    train1 = data.drop(['id', 'price'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(
        train1, labels, test_size=0.10, random_state=2
    )
    clf = GradientBoostingRegressor(
        n_estimators=400, max_depth=5,
        min_samples_split=2, learning_rate=0.1,
        loss='squared_error'
    )
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    return clf, score, train1.columns.tolist()

clf, r2_score, feature_cols = load_model()

# ─── UI ───
st.title("🏠 House Price Predictor")
st.caption("King County, USA — Gradient Boosting Model")

st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("House Details")
    bedrooms   = st.slider("Bedrooms", 1, 10, 3)
    bathrooms  = st.slider("Bathrooms", 1.0, 8.0, 2.0, step=0.25)
    sqft_living = st.slider("Living Area (sqft)", 500, 10000, 2000, step=100)
    sqft_lot   = st.slider("Lot Size (sqft)", 500, 50000, 5000, step=500)
    floors     = st.selectbox("Floors", [1.0, 1.5, 2.0, 2.5, 3.0])
    waterfront = st.radio("Waterfront?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    view       = st.slider("View Quality (0–4)", 0, 4, 0)
    condition  = st.slider("Condition (1–5)", 1, 5, 3)
    grade      = st.slider("Grade (1–13)", 1, 13, 7)

with col2:
    st.subheader("Size & Location")
    sqft_above    = st.slider("Sqft Above Ground", 500, 8000, 1500, step=100)
    sqft_basement = st.slider("Sqft Basement", 0, 3000, 0, step=100)
    yr_built      = st.slider("Year Built", 1900, 2015, 1990)
    yr_renovated  = st.slider("Year Renovated (0 = never)", 0, 2015, 0)
    zipcode       = st.number_input("Zipcode", value=98052)
    lat  = st.number_input("Latitude",  value=47.56, format="%.4f")
    long = st.number_input("Longitude", value=-122.18, format="%.4f")
    sqft_living15 = st.slider("Neighbors Avg Living Sqft", 500, 6000, 1800, step=100)
    sqft_lot15    = st.slider("Neighbors Avg Lot Sqft", 500, 30000, 5000, step=500)

st.markdown("---")

# ─── Predict ───
if st.button("💰 Predict Price", use_container_width=True):
    input_data = pd.DataFrame([{
        'date': 1,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft_living': sqft_living,
        'sqft_lot': sqft_lot,
        'floors': floors,
        'waterfront': waterfront,
        'view': view,
        'condition': condition,
        'grade': grade,
        'sqft_above': sqft_above,
        'sqft_basement': sqft_basement,
        'yr_built': yr_built,
        'yr_renovated': yr_renovated,
        'zipcode': zipcode,
        'lat': lat,
        'long': long,
        'sqft_living15': sqft_living15,
        'sqft_lot15': sqft_lot15
    }])[feature_cols]

    prediction = clf.predict(input_data)[0]
    low  = prediction * 0.85
    high = prediction * 1.15

    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated Price",  f"${prediction:,.0f}")
    c2.metric("Low Estimate",     f"${low:,.0f}")
    c3.metric("High Estimate",    f"${high:,.0f}")

    st.success(f"Model R² Score: **{r2_score:.2f}** (trained on 19,000+ houses)")