import torchvision.models as models
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

import torch.nn as nn
import torchvision.models as models

class CarBikeClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        """

        :param num_classes:
        :param pretrained:
        """
        super(CarBikeClassifier, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        return self.model(x)