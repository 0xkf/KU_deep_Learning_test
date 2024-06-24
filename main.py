# # main.py
# from utils.data_loader import load_data
# from models.neural_network import NeuralNetwork
# import numpy as np

# # ハイパーパラメータの設定
# input_size = 28 * 28
# hidden_size = 128
# output_size = 10
# learning_rate = 0.1
# epochs = 10
# batch_size = 32

# # データの読み込み
# (x_train, y_train), (x_test, y_test) = load_data()

# # ネットワークの初期化
# nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)

# # 学習の実行
# for epoch in range(epochs):
#     for i in range(0, x_train.shape[0], batch_size):
#         x_batch = x_train[i:i + batch_size]
#         y_batch = y_train[i:i + batch_size]
#         nn.train(x_batch, y_batch)
#     print(f"Epoch {epoch + 1}/{epochs} completed")

# # テストデータでの評価
# predictions = nn.predict(x_test)
# accuracy = np.mean(predictions == np.argmax(y_test, axis=1))
# print(f"Accuracy: {accuracy * 100:.2f}%")

from utils.data_loader import load_data
from models.neural_network import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ハイパーパラメータの設定
input_size = 28 * 28
hidden_size = 128
output_size = 10
learning_rate = 0.1
epochs = 10
batch_size = 32

# データの読み込み
(x_train, y_train), (x_test, y_test) = load_data()

# ネットワークの初期化
nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)

# 学習の進行状況を記録するリスト
train_losses = []
train_accuracies = []

# 学習の実行
for epoch in range(epochs):
    epoch_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for i in range(0, x_train.shape[0], batch_size):
        x_batch = x_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]
        nn.train(x_batch, y_batch)

        # 損失計算と精度計算
        output = nn.feedforward(x_batch)
        loss = np.mean((y_batch - output) ** 2)
        epoch_loss += loss

        predictions = np.argmax(output, axis=1)
        correct_predictions += np.sum(predictions == np.argmax(y_batch, axis=1))
        total_predictions += x_batch.shape[0]

    epoch_loss /= (x_train.shape[0] // batch_size)
    train_losses.append(epoch_loss)
    train_accuracy = correct_predictions / total_predictions
    train_accuracies.append(train_accuracy)

    print(f"Epoch {epoch + 1}/{epochs} completed - Loss: {epoch_loss:.4f}, Accuracy: {train_accuracy:.4f}")

# 学習曲線のプロット
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss Over Epochs')

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training Accuracy Over Epochs')

plt.show()

# テストデータでの評価
predictions = nn.predict(x_test)
accuracy = np.mean(predictions == np.argmax(y_test, axis=1))
print(f"Accuracy: {accuracy * 100:.2f}%")

# 混同行列の表示
conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# テストデータのいくつかのサンプルを表示して予測結果を確認
num_samples = 5
indices = np.random.choice(x_test.shape[0], num_samples, replace=False)
for index in indices:
    image = x_test[index].reshape(28, 28)
    true_label = np.argmax(y_test[index])
    predicted_label = predictions[index]

    plt.imshow(image, cmap='gray')
    plt.title(f"True Label: {true_label}, Predicted Label: {predicted_label}")
    plt.show()
