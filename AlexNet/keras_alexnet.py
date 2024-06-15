import keras

INPUT = keras.layers.Input((227, 227, 3))

c1 = keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu')(INPUT)
b1 = keras.layers.BatchNormalization()(c1)
mp = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(b1)

c2 = keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(mp)
b2 = keras.layers.BatchNormalization()(c2)
mp = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(b2)

c3 = keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(mp)
c4 = keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(c3)

c5 = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(c4)
mp = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(c5)

flatten = keras.layers.Flatten()(mp)

fc1 = keras.layers.Dense(4096, activation='relu')(flatten)
fc2 = keras.layers.Dense(4096)(fc1)

OUTPUT = keras.layers.Dense(1000, activation='softmax')(fc2)

MODEL = keras.Model(inputs=[INPUT], outputs=[OUTPUT])
MODEL.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
MODEL.summary()
