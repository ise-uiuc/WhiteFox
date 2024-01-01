
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Conv2d(3, 32, 3, 1, 0), torch.nn.ReLU()
        self.split_branch = torch.nn.Conv2d(32, 16, 5, 2, 1)
    def forward(self, x1):
        split_tensors = torch.split(x1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return concatenated_tensor
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Conv2d(32, 32, 3, 1, 1), torch.nn.ReLU()
        self.concat_branch = torch.nn.Conv2d(32, 64, 5, 2, 1)
    def forward(self, x1):
        split_tensors = torch.split(x1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return concatenated_tensor
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1 = Model1()
        self.branch2 = Model2()
        self.other_features = torch.nn.Sequential(torch.nn.BatchNorm2d(32))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return concatenated_tensor
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
