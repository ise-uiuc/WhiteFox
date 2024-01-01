
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Conv2d(3, 32, 3, 1, 1)
    def forward(self, x):
        y = self.features(x)
        split_tensors_0 = torch.split(y, [1, 4, 1], dim=1)
        split_tensors_1 = torch.split(y, [1, 1], dim=1)
        return (split_tensors_0[0], split_tensors_1[0])
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Conv2d(3, 32, 3, 1, 1)
    def forward(self, x):
        y = self.features(x)
        split_tensors_0 = torch.split(y, [1, 4, 1], dim=1)
        split_tensors_1 = torch.split(y, [1, 1], dim=1)
        return (split_tensors_0[0], split_tensors_1[0])
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
