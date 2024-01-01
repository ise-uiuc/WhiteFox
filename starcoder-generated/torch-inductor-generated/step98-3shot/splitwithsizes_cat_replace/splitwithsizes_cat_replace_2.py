
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ModuleList()
        self.features.append(torch.nn.Sequential(*[torch.nn.ReLU() for _ in range(3)]))
        for i in range(3):
            self.features.append(torch.nn.Conv2d(3, 32, 3, 1, 1, bias=True))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
