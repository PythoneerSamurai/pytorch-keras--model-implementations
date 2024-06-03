import keras

INPUTS = keras.layers.Input((128, 128, 3))
normalized = keras.layers.Lambda(lambda x: x / 255.0)(INPUTS)

# Encoder Path

c1 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same',
                         kernel_initializer=keras.initializers.HeNormal)(normalized)
dp = keras.layers.Dropout(0.1)(c1)
c2 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same',
                         kernel_initializer=keras.initializers.HeNormal)(dp)

mp = keras.layers.MaxPool2D(pool_size=(2, 2))(c2)
c3 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same',
                         kernel_initializer=keras.initializers.HeNormal)(mp)
dp = keras.layers.Dropout(0.1)(c3)
c4 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same',
                         kernel_initializer=keras.initializers.HeNormal)(dp)

mp = keras.layers.MaxPool2D(pool_size=(2, 2))(c4)
c5 = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same',
                         kernel_initializer=keras.initializers.HeNormal)(mp)
dp = keras.layers.Dropout(0.2)(c5)
c6 = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same',
                         kernel_initializer=keras.initializers.HeNormal)(dp)

mp = keras.layers.MaxPool2D(pool_size=(2, 2))(c6)
c7 = keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same',
                         kernel_initializer=keras.initializers.HeNormal)(mp)
dp = keras.layers.Dropout(0.2)(c7)
c8 = keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same',
                         kernel_initializer=keras.initializers.HeNormal)(dp)

mp = keras.layers.MaxPool2D(pool_size=(2, 2))(c8)
c9 = keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), activation='relu', padding='same',
                         kernel_initializer=keras.initializers.HeNormal)(mp)
dp = keras.layers.Dropout(0.3)(c9)
c10 = keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), activation='relu', padding='same',
                          kernel_initializer=keras.initializers.HeNormal)(dp)

# Decoder Path

ct1 = keras.layers.Conv2DTranspose(filters=512, strides=(2, 2), kernel_size=(2, 2),
                                   kernel_initializer=keras.initializers.HeNormal)(c10)
concatenate = keras.layers.concatenate([c8, ct1])
c11 = keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same',
                          kernel_initializer=keras.initializers.HeNormal)(concatenate)
dp = keras.layers.Dropout(0.3)(c11)
c12 = keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same',
                          kernel_initializer=keras.initializers.HeNormal)(dp)

ct2 = keras.layers.Conv2DTranspose(filters=256, strides=(2, 2), kernel_size=(2, 2),
                                   kernel_initializer=keras.initializers.HeNormal)(c12)
concatenate = keras.layers.concatenate([c6, ct2])
c13 = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same',
                          kernel_initializer=keras.initializers.HeNormal)(concatenate)
dp = keras.layers.Dropout(0.2)(c13)
c14 = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same',
                          kernel_initializer=keras.initializers.HeNormal)(dp)

ct3 = keras.layers.Conv2DTranspose(filters=128, strides=(2, 2), kernel_size=(2, 2),
                                   kernel_initializer=keras.initializers.HeNormal)(c14)
concatenate = keras.layers.concatenate([c4, ct3])
c15 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same',
                          kernel_initializer=keras.initializers.HeNormal)(concatenate)
dp = keras.layers.Dropout(0.2)(c15)
c16 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same',
                          kernel_initializer=keras.initializers.HeNormal)(dp)

ct4 = keras.layers.Conv2DTranspose(filters=64, strides=(2, 2), kernel_size=(2, 2),
                                   kernel_initializer=keras.initializers.HeNormal)(c16)
concatenate = keras.layers.concatenate([c2, ct4])
c17 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same',
                          kernel_initializer=keras.initializers.HeNormal)(concatenate)
dp = keras.layers.Dropout(0.1)(c17)
c18 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same',
                          kernel_initializer=keras.initializers.HeNormal)(dp)

OUTPUT = keras.layers.Conv2D(filters=3, kernel_size=(1, 1), activation='sigmoid')(c18)

MODEL = keras.Model(inputs=[INPUTS], outputs=[OUTPUT])

MODEL.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
MODEL.summary()
