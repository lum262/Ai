# Ai
# استيراد المكتبات الأساسية
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# تحميل البيانات
df = pd.read_csv('your_stock_data.csv')  # تأكدي من تحميل الملف في Colab

# تصحيح أسماء الأعمدة (إزالة الفراغات الزائدة)
df.columns = df.columns.str.strip()

# اختيار الأعمدة
features = ['open', 'high', 'low', 'close', 'volume_traded']
data = df[features].copy()

# التعامل مع القيم المفقودة
data.dropna(inplace=True)

# تحجيم البيانات
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# إنشاء التسلسلات الزمنية
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 3])  # عمود "close"
    return np.array(X), np.array(y)

SEQ_LENGTH = 60
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# تقسيم البيانات
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# تحويل البيانات إلى Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# تصميم نموذج RNN
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

model = RNNModel(input_size=5, hidden_size=64, num_layers=2)

# تحديد دالة الخسارة والمُحسن
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# تدريب النموذج
EPOCHS = 50
for epoch in range(EPOCHS):
    model.train()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.6f}")

# اختبار النموذج
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)

# تحويل النتائج إلى NumPy
y_pred = y_pred_tensor.numpy()
y_actual = y_test_tensor.numpy()

# حساب MSE و RMSE
mse = mean_squared_error(y_actual, y_pred)
rmse = np.sqrt(mse)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')

# رسم النتائج
plt.figure(figsize=(12, 6))
plt.plot(y_actual, label='Actual Close Price')
plt.plot(y_pred, label='Predicted Close Price')
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Time')
plt.ylabel('Scaled Price')
plt.legend()
plt.grid()
plt.show()
