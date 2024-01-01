
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        block = [torch.nn.Conv2d(3, 32, 3, 1, 1, bias=False)]
        self.features = torch.nn.Sequential(*block * 3)
    def forward(self, v1, v2, v3):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat([split_tensors[i] * v2[i] for i in range(len(split_tensors))])
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(Model1(), torch.nn.Linear(32, 32), torch.nn.Linear(32, 3))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
