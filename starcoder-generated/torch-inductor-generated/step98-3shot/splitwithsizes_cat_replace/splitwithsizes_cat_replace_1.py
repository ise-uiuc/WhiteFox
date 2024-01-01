
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features1 = torch.nn.ModuleList()
        self.features2 = torch.nn.ModuleList()
        self.features3 = torch.nn.ModuleList()
        for i in range(3):
            self.features1.append(torch.nn.Conv2d(32, 1, 3, 1, 1))
            self.features2.append(torch.nn.Conv2d(32, 1, 3, 1, 1))
            self.features3.append(torch.nn.Conv2d(32, 1, 3, 1, 1))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
