import tensorflow as tf

_models = {}

def get_model(data_type):
    if data_type not in _models:
        if data_type == "tabular":
            model = tf.keras.models.load_model("models/saved/tf_tabular.keras")
        elif data_type == "image":
            model = tf.keras.models.load_model("models/saved/tf_image.keras")
        else:
            model = tf.keras.models.load_model("models/saved/tf_audio.keras")
        _models[data_type] = model
    return _models[data_type]


def invalidate_cache(data_type=None):
    """Elimina del caché el modelo indicado (o todos si no se especifica),
    para forzar que la próxima llamada a get_model() lo recargue desde
    disco. Se debe llamar justo después de guardar un modelo recién
    entrenado, así /predict/ nunca usa pesos viejos en memoria."""
    if data_type is None:
        _models.clear()
    else:
        _models.pop(data_type, None)