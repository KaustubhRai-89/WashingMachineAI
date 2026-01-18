import tensorflow as tf
from tensorflow.keras import layers, models, Model


def build_washing_machine_model(input_shape=(224, 224, 3),
                                num_fabrics=5,
                                num_dirt_types=5):
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = False

    inputs = base_model.input
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    fabric_out = layers.Dense(num_fabrics, activation='softmax', name='fabric_output')(x)

    dirt_type_out = layers.Dense(num_dirt_types, activation='softmax', name='dirt_output')(x)

    intensity_out = layers.Dense(1, activation='sigmoid', name='intensity_output')(x)

    model = Model(inputs=inputs,
                  outputs=[fabric_out, dirt_type_out, intensity_out])

    return model


if __name__ == "__main__":
    model = build_washing_machine_model()
    model.summary()
    print("Model built successfully.")