
import torch
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.features = torch.nn.Sequential(

        )

    def forward(self, x):
        feature_out = self.features(x)
        out = torch.tanh(feature_out)
        return out
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
