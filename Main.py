import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

X, y = make_moons(n_samples=10000, noise=0.2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(2,))
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=16,
                    validation_split=0.1,
                    verbose=0)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.2f}")


sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, palette="coolwarm", s=50)
plt.title("Two Moons Dataset")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

#Accuracy and Loss curves over Epochs
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.title("Model Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Model Loss")
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

#sns.heatmap(x=X[:,0], y=X[:,1], hue=y, palette="coolwarm", linewidth=0.5, square=True)
#Accuracy at the end of training
y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
plt.scatter(X_test[:, 0], X_test[:, 1], c=(y_test == y_pred), cmap='coolwarm')
plt.title("Correct (Red) vs Incorrect (Blue) Predictions")
plt.legend(loc='upper left')
plt.show()

