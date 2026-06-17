import torch.nn as nn
import torchvision.models as models

def get_resnet_model():

    # Step 1: Load pretrained ResNet50
    model = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT
    )

    # Step 2: Remove final FC layer
    model = nn.Sequential(*list(model.children())[:-1])

    model.eval()

    return model