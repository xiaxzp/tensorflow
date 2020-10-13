import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
print(x_train.dtype)
print(y_train.dtype)
print(y_train.size)
x_train, x_test = tf.reshape(x_train / 255.0, [-1,28,28,1]), tf.reshape(x_test / 255.0, [-1,28,28,1])
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(8, 5, activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Conv2D(4, 3, activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Flatten(input_shape=(5, 5)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=2)
model.evaluate(x_test, y_test)
