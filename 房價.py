from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# 只選擇幾個簡單、對高中生容易理解的欄位作為特徵
features = ['行政區', '建物總面積', '屋齡', '房數', '廳數', '衛數', '電梯']
target = '總價'

# 複製資料並轉換行政區為數值
data = df[features + [target]].copy()
le = LabelEncoder()
data['行政區'] = le.fit_transform(data['行政區'])

# 分割訓練集與測試集
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立並訓練線性回歸模型
model = LinearRegression()
model.fit(X_train, y_train)

# 預測與評估模型表現
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

model.coef_, model.intercept_, mae
