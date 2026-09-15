import torch

_models = {}

def get_model(data_type):
    if data_type not in _models:
        # Importar tu clase de modelo según el tipo
        if data_type == "tabular":
            from .pytorch_arch import TabularNet
            model = TabularNet()
            model.load_state_dict(
                torch.load("models/saved/pt_tabular.pt", map_location="cpu")
                )
        elif data_type == "image":
            from .pytorch_arch import ImageCNN
            model = ImageCNN()
            model.load_state_dict(torch.load("models/saved/pt_image.pt", map_location="cpu"))
        else:
            from .pytorch_arch import AudioCNN
            model = AudioCNN()
            model.load_state_dict(torch.load("models/saved/pt_audio.pt", map_location="cpu" ))
        model.eval()
        _models[data_type] = model
    return _models[data_type]


def invalidate_cache(data_type=None):
    if data_type is None:
        _models.clear()
    else:
        _models.pop(data_type, None)