import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# === Load model ===
with open("catboost_model.sav", "rb") as file:
    model = pickle.load(file)

# === Build encoders ===
def build_label_encoders():
    encoders = {}

    encoders['HomePlanet'] = LabelEncoder().fit(['Earth', 'Europa', 'Mars', 'nan'])
    encoders['CryoSleep'] = LabelEncoder().fit(['False', 'True'])
    encoders['Cabin'] = LabelEncoder().fit(['B/0/P', 'B/0/S', 'B/100/S', 'F/1/S', 'nan'])
    encoders['Destination'] = LabelEncoder().fit(['55 Cancri e', 'PSO J318.5-22', 'TRAPPIST-1e', 'nan'])
    encoders['VIP'] = LabelEncoder().fit(['False', 'True'])

    return encoders

encoders = build_label_encoders()

# === Encode inputs ===
def encode_input(raw_input, encoders):
    encoded = {}
    for key, value in raw_input.items():
        if key in encoders:
            val_str = str(value)
            encoder = encoders[key]
            encoded[key] = encoder.transform([val_str])[0] if val_str in encoder.classes_ else encoder.transform(['nan'])[0]
        else:
            encoded[key] = value
    return encoded

# === Streamlit UI ===
st.set_page_config(page_title="CatBoost Transport Predictor", layout="centered")
st.title("🚀 Passenger Transport Predictor")
st.markdown("Fill in passenger details to predict if they were transported!")

with st.form("prediction_form"):
    home_planet = st.selectbox("HomePlanet", ["Earth", "Europa", "Mars"])
    cryo_sleep = st.radio("CryoSleep", ["yes", "no"])
    cabin = st.text_input("Cabin (e.g., F/1/S)", "F/1/S")
    destination = st.selectbox("Destination", ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"])
    age = st.number_input("Age", min_value=0.0, max_value=100.0, step=1.0)
    vip = st.radio("VIP", ["yes", "no"])

    room_service = st.number_input("RoomService", min_value=0.0, step=1.0)
    food_court = st.number_input("FoodCourt", min_value=0.0, step=1.0)
    shopping_mall = st.number_input("ShoppingMall", min_value=0.0, step=1.0)
    spa = st.number_input("Spa", min_value=0.0, step=1.0)
    vr_deck = st.number_input("VRDeck", min_value=0.0, step=1.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_dict = {
        "HomePlanet": home_planet,
        "CryoSleep": "True" if cryo_sleep == "yes" else "False",
        "Cabin": cabin,
        "Destination": destination,
        "Age": age,
        "VIP": "True" if vip == "yes" else "False",
        "RoomService": room_service,
        "FoodCourt": food_court,
        "ShoppingMall": shopping_mall,
        "Spa": spa,
        "VRDeck": vr_deck
    }

    encoded_input = encode_input(input_dict, encoders)
    df = pd.DataFrame([encoded_input])
    prediction = model.predict(df)[0]

    st.subheader("🎯 Prediction Result:")
    if prediction:
        st.success("✅ The Passenger was TRANSPORTED!")
    else:
        st.error("❌ The Passenger was NOT transported.")
