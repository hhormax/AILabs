import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. загрузка датасета
df = pd.read_csv("new_test.csv")

print("размер датасета:", df.shape)
print(df.head())


# 2. задача регрессии
# предсказываем непрерывный признак Age (возраст)
X = df.drop(["Age"], axis=1)
y = df["Age"]

# делим данные на обучающую (80%) и тестовую (20%) выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# создаём и обучаем модель линейной регрессии
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# делаем предсказание для тестовой выборки
y_pred_test = linear_model.predict(X_test)


# 3. оценка регрессионной модели
MSE = mean_squared_error(y_test, y_pred_test)
RMSE = root_mean_squared_error(y_test, y_pred_test)
MAE = mean_absolute_error(y_test, y_pred_test)

print("\nоценка регрессионной модели:")
print(f"MSE  = {MSE:.4f}")
print(f"RMSE = {RMSE:.4f}")
print(f"MAE  = {MAE:.4f}")

# Age был нормализован в первой лабораторной работе, поэтому ошибки
# тоже находятся в диапазоне от 0 до 1. MAE примерно 0.13 — результат нормальный.


# 4. задача классификации
# предсказываем категориальный признак CryoSleep (0 или 1)
X = df.drop(["CryoSleep"], axis=1)
y = df["CryoSleep"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# создаём и обучаем модель логистической регрессии
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)

y_pred_test = logreg_model.predict(X_test)


# 5. оценка классификационной модели
accuracy = accuracy_score(y_test, y_pred_test)
cm = confusion_matrix(y_test, y_pred_test)
report = classification_report(y_test, y_pred_test)

print("\nоценка классификационной модели:")
print(f"accuracy = {accuracy:.4f}")

print("\nматрица ошибок:")
print(cm)

print("\nотчёт по классам:")
print(report)
