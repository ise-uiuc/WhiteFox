
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layerA = torch.nn.Conv2d(3, 10, 5)
        self.layerB = torch.nn.Sequential(*(torch.nn.MaxPool2d(5, 1, 2) for _ in range(31)))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        split_tensors_layerB = torch.split(concatenated_tensor, [1, 1, 1], dim=1)
        layer_output = []
        for v in split_tensors_layerB:
            layer_output.append(self.layerA(v))
        layer_output = torch.cat(layer_output, dim=1)
        return (layer_output, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
