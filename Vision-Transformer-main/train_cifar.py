import tensorflow as tf
from util import get_image

from model import ViT
from trainer import Trainer, TrainerConfig

train_images, train_labels, test_images, test_labels = get_image()

train_images = tf.cast(tf.reshape(train_images, (-1, 3, 32, 32)),dtype=tf.float32)

test_images = tf.cast(tf.reshape(test_images,(-1, 3, 32, 32)),dtype=tf.float32)
train_images, test_images = train_images / 255.0, test_images / 255.0

# data_augmentation = tf.keras.Sequential(
#     [tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomRotation(0.1),]
# )
# train_images = data_augmentation(train_images)
# test_images = data_augmentation(test_images)

train_x = tf.data.Dataset.from_tensor_slices(train_images,)
train_y = tf.data.Dataset.from_tensor_slices(train_labels)
train_dataset = tf.data.Dataset.zip((train_x,train_y))
test_x = tf.data.Dataset.from_tensor_slices(test_images)
test_y = tf.data.Dataset.from_tensor_slices(test_labels)
test_dataset = tf.data.Dataset.zip((test_x,test_y))

tconf = TrainerConfig(max_epochs=30, batch_size=60, learning_rate=1e-3, ckpt_path="vit_model")
# sample model config.
model_config = {"image_size":32,
                "patch_size":4,
                "num_classes":100,
                "dim":64,
                "depth":5,
                "heads":4,
                "mlp_dim":500}

trainer = Trainer(ViT, model_config, train_dataset, len(train_images), test_dataset, len(test_images), tconf)

trainer.train()