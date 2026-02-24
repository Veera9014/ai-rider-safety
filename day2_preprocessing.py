import pandas as pd
import joblib

# Load dataset
data = pd.read_csv("rider_safety_data.csv")

print("Original Data:")
print(data.head())

# Convert traffic column into numbers
traffic_mapping = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

data["traffic"] = data["traffic"].map(traffic_mapping)

# Convert road condition into numbers
road_mapping = {
    "Good": 1,
    "Bad": 0
}

data["road_condition"] = data["road_condition"].map(road_mapping)

# Convert risk column into numbers
risk_mapping = {
    "SAFE": 0,
    "WARNING": 1,
    "DANGER": 2
}

data["risk"] = data["risk"].map(risk_mapping)

print("\nConverted Data:")
print(data.head())
x = data.drop("risk",axis=1)
y = data["risk"]
print("\nFeature (x):")
print(x.head())
print("\n target (y):")
print(y.head())
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
print("\nTraining Data Size:",x_train.shape)
print("Testing Data Size:",x_test.shape)
from sklearn.tree import DecisionTreeClassifier
# create model
model= DecisionTreeClassifier()
#Train model
model.fit(x_train,y_train)
#Predict
predictions=model.predict(x_test)

print("\nPredictions:")
print(predictions)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,predictions)
print("\nModel Accuracy:",accuracy)

from sklearn.metrics import confusion_matrix,classification_report

print("\nConfusion Matrix:")
print(confusion_matrix(y_test,predictions))

print("\nClassification Repor:")
print(classification_report(y_test,predictions))

speed=float(input("Enter Speed: "))
distance = float(input("Enter Distance: "))
traffic = int(input("Traffic(Low=0, Medium=1,High=2): "))
road = int(input("Road Condition (Good=1, Bad=0): "))

user_data = [[speed, distance, traffic, road]]

result = model.predict(user_data)

print("\nPredicted Risk Level:", result[0])
joblib.dump(model,"rider_risk_model.pkl")
print("Model saved successfully")
