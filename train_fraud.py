import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

def train_and_save_fraud_model():

    print("\n🚀 Starting Fraud Model Training...")

    # Load dataset
    df = pd.read_csv(r"C:\Users\Abhirami R\Desktop\fraud_data\Fraud.csv")
    print("📄 Dataset Loaded Successfully!")
    print("Shape:", df.shape)

    # Encode categorical column "type"
    le = LabelEncoder()
    df["type"] = le.fit_transform(df["type"]) 
    print("🔤 Encoding Completed!")

    # Select features & target
    X = df[['step','type','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']]
    y = df['isFraud']

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # Save model + scaler + encoder
    with open("fraud_model.pkl", "wb") as f:
        pickle.dump({
            "model": model,
            "scaler": scaler,
            "encoder": le
        }, f)

    print("🎉 Model Saved as fraud_model.pkl Successfully!")

    # Show accuracy
    accuracy = model.score(X_test, y_test)
    print("🔍 Accuracy:", accuracy)


if __name__ == "__main__":
    train_and_save_fraud_model()
