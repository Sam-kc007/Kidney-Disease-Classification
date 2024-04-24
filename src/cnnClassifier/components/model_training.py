import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from cnnClassifier.entity.config_entity import TrainingConfig
from pathlib import Path



class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.gpu_available = tf.config.list_physical_devices("GPU")

    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)

    def train_valid_generator(self):
        datagenerator_kwargs = dict(rescale=1.0 / 255, validation_split=0.20)

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs,
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs,
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs,
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train_model(self):
        self.steps_per_epoch = (
            self.train_generator.samples // self.train_generator.batch_size
        )
        self.validation_steps = (
            self.valid_generator.samples // self.valid_generator.batch_size
        )

        start_time = time.time()
        for epoch in range(self.config.params_epochs):
            print(f"Epoch {epoch + 1}/{self.config.params_epochs}")

            # Training loop
            epoch_start_time = time.time()
            self.model.fit(
                self.train_generator,
                steps_per_epoch=self.steps_per_epoch,
                validation_steps=self.validation_steps,
                validation_data=self.valid_generator,
                verbose=2,
            )
            epoch_end_time = time.time()

            epoch_time = epoch_end_time - epoch_start_time
            print(f"Epoch {epoch + 1} took {epoch_time:.2f} seconds")

        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")

    def train(self):
        if self.gpu_available:
            print("GPU available. Using GPU...")
            with tf.device("/GPU:0"):
                self.train_model()
        else:
            print("No GPU available. Using CPU...")
            self.train_model()

        self.save_model(path=self.config.trained_model_path, model=self.model)
