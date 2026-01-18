import os
import yaml
import sqlite3
import numpy as np
import tensorflow as tf
import cv2
from src.model_architecture import build_washing_machine_model
from tensorflow.keras.utils import to_categorical

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


class LaundryDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, db_path, batch_size=32, image_size=(224, 224), subset="train"):
        self.batch_size = batch_size
        self.image_size = image_size
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.data = self._fetch_data()
        split_idx = int(len(self.data) * 0.8)
        if subset == "train":
            self.data = self.data[:split_idx]
        else:
            self.data = self.data[split_idx:]
        self.indices = np.arange(len(self.data))
        self.fabrics = config["FABRIC_CLASSES"]
        self.dirts = config["DIRT_CLASSES"]

    def _fetch_data(self):
        query = "SELECT filename, fabric_label, dirt_label, dirt_intensity FROM dataset"
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_x = np.zeros((self.batch_size, *self.image_size, 3), dtype=np.float32)
        y_fabric = np.zeros((self.batch_size, len(self.fabrics)), dtype=np.float32)
        y_dirt = np.zeros((self.batch_size, len(self.dirts)), dtype=np.float32)
        y_int = np.zeros((self.batch_size,), dtype=np.float32)

        for i, idx in enumerate(batch_indices):
            filename, fabric_txt, dirt_txt, intensity_val = self.data[idx]
            if "dirty" in filename:
                path = os.path.join(config["SYNTHETIC_DIR"], filename)
            else:
                path = os.path.join(config["PROCESSED_DATA_DIR"], filename)

            image = cv2.imread(path)
            if image is not None:
                image = cv2.resize(image, self.image_size)
                image = image / 255.0
                batch_x[i] = image

            f_idx = self.fabrics.index(fabric_txt) if fabric_txt in self.fabrics else 0
            y_fabric[i] = to_categorical(f_idx, num_classes=len(self.fabrics))

            d_idx = self.dirts.index(dirt_txt) if dirt_txt in self.dirts else 0
            y_dirt[i] = to_categorical(d_idx, num_classes=len(self.dirts))

            y_int[i] = intensity_val

        return batch_x, {"fabric_output": y_fabric, "dirt_output": y_dirt, "intensity_output": y_int}


def train():
    train_gen = LaundryDataGenerator(config["DB_PATH"], subset="train")
    val_gen = LaundryDataGenerator(config["DB_PATH"], subset="val")

    if len(train_gen) == 0:
        print("Error: No data found in database. Run src/data_loader.py first!")
        return

    model = build_washing_machine_model(
        num_fabrics=len(config["FABRIC_CLASSES"]),
        num_dirt_types=len(config["DIRT_CLASSES"])
    )

    losses = {
        "fabric_output": "categorical_crossentropy",
        "dirt_output": "categorical_crossentropy",
        "intensity_output": "mse"
    }

    loss_weights = {"fabric_output": 1.0, "dirt_output": 1.0, "intensity_output": 0.5}

    model.compile(optimizer='adam', loss=losses, loss_weights=loss_weights, metrics=['accuracy'])

    checkpoint_path = os.path.join("models", "checkpoints", "model_epoch_{epoch:02d}.keras")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, save_best_only=True, monitor='val_loss'
    )

    print("Starting Training...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config["EPOCHS"],
        callbacks=[checkpoint]
    )

    model.save("models/final_model.keras")
    print("Training Complete. Model saved.")


if __name__ == "__main__":
    train()