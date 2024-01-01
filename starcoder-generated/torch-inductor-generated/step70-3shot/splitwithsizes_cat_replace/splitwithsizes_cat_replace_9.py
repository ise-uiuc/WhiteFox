
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layers = [torch.nn.Linear(64, 32)]
        layers.append(torch.nn.ReLU6(inplace=True))
        layers.append(torch.nn.Linear(32, 8))
        self.features = torch.nn.Sequential(*layers)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
