import pickle
import numpy as np
import pandas as pd
from catboost import Pool
from sklearn.preprocessing import LabelEncoder

# Load model from file
with open("catboost_model.sav", 'rb') as file:
    model = pickle.load(file)

# === Training-based LabelEncoder setup ===
def build_label_encoders():
    encoders = {}

    # Hardcoded from training CSV
    cryosleep_classes = ['False', 'True']
    cabin_classes = [  # Trimmed example — add more if needed or load from training data
        'B/0/P', 'B/0/S', 'B/100/S', 'F/1/S', 'nan'
    ]
    destination_classes = ['55 Cancri e', 'PSO J318.5-22', 'TRAPPIST-1e', 'nan']
    vip_classes = ['False', 'True']

    def make_encoder(classes):
        le = LabelEncoder()
        le.fit(classes)
        return le
    
    encoders['HomePlanet'] = LabelEncoder().fit(['Earth', 'Europa', 'Mars', 'nan'])
    encoders['CryoSleep'] = make_encoder(cryosleep_classes)
    encoders['Cabin'] = make_encoder(cabin_classes)
    encoders['Destination'] = make_encoder(destination_classes)
    encoders['VIP'] = make_encoder(vip_classes)

    return encoders

encoders = build_label_encoders()

# === Encoding logic ===
def encode_input(raw_input, encoders):
    encoded = {}
    for key, value in raw_input.items():
        if key in encoders:
            val_str = str(value)
            encoder = encoders[key]
            if val_str in encoder.classes_:
                encoded[key] = encoder.transform([val_str])[0]
            else:
                encoded[key] = encoder.transform(['nan'])[0]
        else:
            encoded[key] = value
    return encoded

# === Input handling ===
def get_string(prompt, options=None):
    while True:
        val = input(prompt).strip()
        if options and val not in options:
            print(f"Invalid. Choose from {options}")
        else:
            return val

def get_yes_no(prompt):
    while True:
        val = input(prompt + " (yes/no): ").strip().lower()
        if val in ["yes", "no"]:
            return 'True' if val == 'yes' else 'False'
        print("Enter yes or no.")

def get_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid number. Try again.")

# === Main input ===
def collect_user_input():
    print("\n📋 Enter passenger details:\n")
    data = {
        "HomePlanet": get_string("HomePlanet (Europa/Earth/Mars): ", ["Europa", "Earth", "Mars"]),
        "CryoSleep": get_yes_no("Was the passenger in CryoSleep?"),
        "Cabin": get_string("Cabin (e.g., B/0/P, F/1/S): "),
        "Destination": get_string("Destination (TRAPPIST-1e/55 Cancri e/PSO J318.5-22): ",
                                  ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"]),
        "Age": get_float("Age: "),
        "VIP": get_yes_no("Is the passenger a VIP?"),
        "RoomService": get_float("RoomService spending: "),
        "FoodCourt": get_float("FoodCourt spending: "),
        "ShoppingMall": get_float("ShoppingMall spending: "),
        "Spa": get_float("Spa spending: "),
        "VRDeck": get_float("VRDeck spending: ")
    }
    return data

# === Main ===
if __name__ == "__main__":
    raw_input_data = collect_user_input()
    encoded_data = encode_input(raw_input_data, encoders)

    # Convert to DataFrame
    df = pd.DataFrame([encoded_data])

    # Predict
    prediction = model.predict(df)[0]
    print("\n🚀 Prediction Result 🚀")
    print("✅ The Passenger was TRANSPORTED!" if prediction else "❌ The Passenger was NOT transported.")
