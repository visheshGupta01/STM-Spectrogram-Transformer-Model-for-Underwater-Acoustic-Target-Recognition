import torch
from torch import nn
from torchvision.models import resnet18

class PaperCNN(nn.Module):
    """Two-convolution baseline following the paper; dense dimensions are implementation-defined."""
    def __init__(self, num_classes=5, dropout=0.1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 256, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256 * 8 * 8, 512),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def build_model(name="resnet18", num_classes=5):
    if name == "cnn":
        return PaperCNN(num_classes=num_classes)
    if name == "resnet18":
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError("model must be cnn or resnet18")
