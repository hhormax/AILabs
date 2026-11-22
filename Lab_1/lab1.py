import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1.загрузка данных
df = pd.read_csv("test.csv")

df.head()
cols = df.columns

obj = "Name"
if obj in cols:
    print("Dataset has column Name")
obj = "Drowned"
if obj not in cols:
    print("Dataset hasn’t column Drowned")

# 2.вывод на экран
print("\nпервые 5 строк:")
print(df.head())

print("\nпоследние 5 строк:")
print(df.tail())

print("\nтипы данных по столбцам:")
print(df.dtypes)

print("\nобщая информация о датасете:")
df.info()

# 3.пропущенные значения в каждом столбце
nan = df.isnull()

print("\nпропуски по столбцам:")
missing = nan.sum()
print(missing)
print(f"всего пропущено: {missing.sum()}")

# 4.заполение пропущенного
categorical_cols = ["HomePlanet", "CryoSleep", "Cabin", "Destination", "VIP", "Name"]
mean_cols = ["Age"]
median_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
numeric_cols = mean_cols + median_cols

print("\nзаполнение категориальных признаков модой:")
for col in categorical_cols:
    mode_value = df[col].mode()[0]
    df[col] = df[col].fillna(mode_value)
    print(f"  {col}: мода = {mode_value}")

print("\nзаполнение трат медианой:")
for col in median_cols:
    median_value = df[col].median()
    df[col] = df[col].fillna(median_value)
    print(f"  {col}: медиана = {median_value:.2f}")

print("\nзаполнение Age средним значением:")
for col in mean_cols:
    mean_value = df[col].mean()
    df[col] = df[col].fillna(mean_value)
    print(f"  {col}: среднее = {mean_value:.2f}")

print("\nПропуски до и после заполнения:")
missing_after = df.isnull().sum()
print(pd.DataFrame({"до": missing, "после": missing_after}))
print(f"Осталось пропущенных значений: {missing_after.sum()}")

# 5.нормализация
print("\nдо нормализации:")
print(df[numeric_cols].describe().loc[["min", "mean", "max"]])

raw_numeric = df[numeric_cols].copy()
minmax_scaler = MinMaxScaler()
df[numeric_cols] = minmax_scaler.fit_transform(df[numeric_cols])

print("\nMinMax-нормализация:")
print(df[numeric_cols].describe().loc[["min", "mean", "max"]])

df_standard = raw_numeric.copy()
standard_scaler = StandardScaler()
df_standard[numeric_cols] = standard_scaler.fit_transform(df_standard[numeric_cols])
print("\nнормализация StandardScaler (Z-оценка):")
print(df_standard[numeric_cols].describe().loc[["mean", "std"]])

# 6.категориальные данные в численные
# в Cabin много уникальных значений, one-hot encoding нецелесообразен(переобучение)
cabin_parts = df["Cabin"].str.split("/", expand=True)
df["Deck"] = cabin_parts[0]
df["Side"] = cabin_parts[2]

df = df.drop(columns=["PassengerId", "Name", "Cabin"])
df["CryoSleep"] = df["CryoSleep"].astype(int)
df["VIP"] = df["VIP"].astype(int)

categorical_to_encode = ["HomePlanet", "Destination", "Deck", "Side"]
df = pd.get_dummies(df, columns=categorical_to_encode, drop_first=True, dtype=int)

print("\nитоговый датасет:")
print(df.head())

df.to_csv("new_test.csv", index=False)
