# -
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# 1. 讀取資料
df = pd.read_csv("Taipei_house.csv")

# 2. 建立「是否有車位」欄位：有車位 = 1，無車位 = 0
df['有車位'] = df['車位類別'].apply(lambda x: 0 if x.strip() == '無' else 1)

# 3. 把「行政區」轉換為數字（Label Encoding）
le = LabelEncoder()
df['行政區'] = le.fit_transform(df['行政區'])

# 4. 選擇用來預測的欄位（特徵）
features = ['行政區', '建物總面積', '屋齡', '房數', '廳數', '衛數', '電梯', '有車位']
target = '總價'

# 5. 準備資料
X = df[features]
y = df[target]

# 6. 分割訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. 建立並訓練線性回歸模型
model = LinearRegression()
model.fit(X_train, y_train)

# 8. 預測與評估模型
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

# 9. 顯示結果
print("模型係數（每個欄位對價格的影響）:")
for name, coef in zip(features, model.coef_):
    print(f"{name}: {coef:.2f}")

print(f"\n預測誤差（平均差多少）：{mae:.0f} 萬元")
