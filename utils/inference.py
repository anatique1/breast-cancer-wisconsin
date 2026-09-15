import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def run_inference(x, framework, data_type):
    if framework == "tensorflow":
        from models.tensorflow_models import get_model
        model = get_model(data_type)
        # TensorFlow ya devuelve probabilidades gracias a la activación 'softmax' final
        preds = model.predict(x)
    else:
        from models.pytorch_models import get_model
        model = get_model(data_type)
        import torch
        import torch.nn.functional as F
        
        # Nos aseguramos de que los datos sean un Tensor antes de pasarlos a PyTorch
        if not isinstance(x, torch.Tensor):
            # Si es un DataFrame de Pandas, tomamos los .values
            if hasattr(x, 'values'):
                x = torch.tensor(x.values, dtype=torch.float32)
            else:
                x = torch.tensor(x, dtype=torch.float32)
                
        with torch.no_grad():
            out = model(x)
            # PyTorch devuelve logits crudos (porque usas CrossEntropyLoss).
            # Para la confianza, necesitamos aplicar softmax y convertirlos a probabilidades (0 a 1)
            if out.shape[1] > 1:
                preds = F.softmax(out, dim=1).cpu().numpy()
            else:
                preds = torch.sigmoid(out).cpu().numpy()

    # Convertir probabilidades a clases y obtener la confianza máxima
    if data_type == "tabular":
        if preds.shape[1] == 1:
            # Clasificación binaria (1 solo nodo con sigmoid)
            classes = (preds > 0.5).astype(int).flatten()
            # La confianza es 'p' si es 1, o '1-p' si es 0
            confidences = np.where(preds > 0.5, preds, 1 - preds).flatten()
        else:
            # Multiclase (2 o más nodos con softmax)
            classes = np.argmax(preds, axis=1)
            confidences = np.max(preds, axis=1)
            
        return classes.tolist(), confidences.tolist()
    
    # Para imágenes o audio
    classes = np.argmax(preds, axis=1)
    confidences = np.max(preds, axis=1)
    return classes.tolist(), confidences.tolist()