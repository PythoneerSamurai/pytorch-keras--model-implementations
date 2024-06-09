import keras

INPUT = keras.layers.Input((224, 224, 3))

c1 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(INPUT)
c2 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(c1)

mp = keras.layers.MaxPooling2D(pool_size=(2, 2))(c2)

c3 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(mp)
c4 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(c3)

mp = keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

c5 = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(mp)
c6 = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(c5)
c7 = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(c6)
c8 = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(c7)

mp = keras.layers.MaxPooling2D(pool_size=(2, 2))(c8)

c9 = keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(mp)
c10 = keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(c9)
c11 = keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(c10)
c12 = keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(c11)

mp = keras.layers.MaxPooling2D(pool_size=(2, 2))(c12)

c13 = keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(mp)
c14 = keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(c13)
c15 = keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(c14)
c16 = keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(c15)

mp = keras.layers.MaxPooling2D(pool_size=(2, 2))(c16)

flatten = keras.layers.Flatten()(mp)

fc1 = keras.layers.Dense(4096)(flatten)
fc2 = keras.layers.Dense(4096)(fc1)
fc3 = keras.layers.Dense(1000)(fc2)

OUTPUT = fc3

MODEL = keras.Model(inputs=[INPUT], outputs=[OUTPUT])
MODEL.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
MODEL.summary()
