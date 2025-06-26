import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# מייבא את הפונקציה מהקובץ הקודם
from assets_data_prep import prepare_data

# טוען את הנתונים מהקובץ של חלק ב
df = pd.read_csv("train.csv")

# מכין את הנתונים בעזרת הפונקציה
processed_df = prepare_data(df, dataset_type="train")

# מפריד בין הפיצ'רים למשתנה המטרה (המחיר)
X = processed_df.drop("price", axis=1)
y = processed_df["price"]

# מחלקים ל-Train ו-Test כדי לבדוק ביצועים (לא חובה אבל טוב שיהיה)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# מאמן את המודל (אפשר להחליף ל-ElasticNet אם רוצים)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# שומר את המודל המאומן לקובץ
joblib.dump(model, "trained_model.pkl")

print("✅ Model trained and saved as trained_model.pkl")
