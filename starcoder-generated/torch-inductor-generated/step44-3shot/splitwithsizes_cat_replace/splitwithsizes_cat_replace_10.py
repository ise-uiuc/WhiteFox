
class Layer1(torch.nn.Module):
    def __init__(self, inp, hidden):
        super().__init__()
        self.fc1 = torch.nn.Linear(inp, hidden)
        self.bn1 = torch.nn.BatchNorm1d(hidden, affine=False, track_running_stats=False)
    def forward(self, v1):
        return self.bn1(self.fc1(v1))
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            Layer1(3, 16),
            Layer1(32, 64),
            Layer1(128, 96)
        ])
        self.features = torch.cat([self.layers[0](v1), self.layers[1](v1), self.layers[2](v1)], dim=1)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
