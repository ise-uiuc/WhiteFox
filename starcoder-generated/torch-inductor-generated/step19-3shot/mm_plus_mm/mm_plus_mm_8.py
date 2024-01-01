
import torch.nn as nn
class Model(nn.Module):
    def __init__(self, in_size, h1_size, h2_size, num_classes):
        super(Model, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_size, h1_size),
            nn.ReLU(),
            nn.Linear(h1_size, h2_size),
            nn.ReLU(),
            nn.Linear(h2_size, num_classes)
        )


    def forward(self, input):
        return self.classifier(input)
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(4, 2)
