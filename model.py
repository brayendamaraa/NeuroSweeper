import tensorflow as tf
from tensorflow.keras import models, layers

def build_model():
    model = models.Sequential([
        layers.Input(shape=(None, None, 11)),

        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.Conv2D(64, 3, padding='same', activation='relu'),

        # output: mine probability per cell
        layers.Conv2D(1, 1, padding='same', activation='tanh')
    ])
    return model

def wmse(y_true, y_pred):
    # Tentukan bobot untuk sel yang berisi ranjau (1)
    # Misalnya, memberi bobot 50x lebih besar daripada sel tanpa ranjau (0)
    mine_weight = 500
    
    # 1. Hitung error kuadrat dasar
    squared_errors = tf.square(y_true - y_pred)
    
    # 2. Buat matriks bobot:
    # Jika y_true == 1 atau y_true == -1, berikan bobot mine_weight. Jika y_true == 0, beri bobot 1.0.
    weights = tf.where(tf.logical_or(tf.equal(y_true, 1), tf.equal(y_true, -1)),
                       tf.cast(mine_weight, tf.float32),
                       tf.cast(1.0, tf.float32))
    
    # 3. Kalikan error dengan bobot
    weighted_errors = squared_errors * weights
    
    # 4. Kembalikan rata-rata dari seluruh error berbobot
    return tf.reduce_mean(weighted_errors)