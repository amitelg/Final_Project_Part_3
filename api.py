from flask import Flask, request, render_template
import pandas as pd
import joblib

from assets_data_prep import prepare_data  # פונקציית ההכנה

app = Flask(__name__)

# טוען את המודל המאומן
model = joblib.load("trained_model.pkl")

@app.route('/')
def index():
    return render_template("index.html")  # מציג את טופס הקלט

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # קבלת הנתונים מהטופס
        input_data = {
            "room_num": float(request.form["room_num"]),
            "area": float(request.form["area"]),
            "floor": request.form["floor"],
            "total_floors": request.form["total_floors"],
            "monthly_arnona": float(request.form["monthly_arnona"]),
            "building_tax": float(request.form["building_tax"]),
            "garden_area": float(request.form["garden_area"]),
            "days_to_enter": float(request.form["days_to_enter"]),
            "num_of_payments": float(request.form["num_of_payments"]),
            "num_of_images": float(request.form["num_of_images"]),
            "distance_from_center": float(request.form["distance_from_center"]),
            "has_parking": 1 if request.form.get("has_parking") == "on" else 0,
            "has_storage": 1 if request.form.get("has_storage") == "on" else 0,
            "elevator": 1 if request.form.get("elevator") == "on" else 0,
            "ac": 1 if request.form.get("ac") == "on" else 0,
            "handicap": 1 if request.form.get("handicap") == "on" else 0,
            "has_bars": 1 if request.form.get("has_bars") == "on" else 0,
            "has_safe_room": 1 if request.form.get("has_safe_room") == "on" else 0,
            "has_balcony": 1 if request.form.get("has_balcony") == "on" else 0,
            "is_furnished": 1 if request.form.get("is_furnished") == "on" else 0,
            "is_renovated": 1 if request.form.get("is_renovated") == "on" else 0,
            "neighborhood": request.form["neighborhood"],
            "property_type": request.form["property_type"]
        }

        # הפיכת המידע ל-DataFrame
        input_df = pd.DataFrame([input_data])

        # הכנת הקלט
        processed_input = prepare_data(input_df, dataset_type="test")

        # חיזוי
        prediction = model.predict(processed_input)[0]
        prediction = round(prediction, 2)

        return render_template("index.html", prediction=prediction)

    except Exception as e:
        return f"❌ Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
