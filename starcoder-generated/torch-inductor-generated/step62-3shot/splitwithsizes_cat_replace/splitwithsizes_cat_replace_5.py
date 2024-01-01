
class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 64, 3, 1, 0, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return torch.nn.ReLU()(self.bn1(self.conv1(concatenated_tensor)))
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = [torch.nn.Conv2d(3, 32, 3, 1, 1, bias=False)]
        self.features_1 = [torch.nn.Conv2d(32, 32, 3, 1, 1, bias=False)]
        self.features_2 = [torch.nn.BatchNorm2d(32)]
        self.features_3 = [torch.nn.ReLU()]
        self.features_4 = [Block()]
        self.features_5 = [torch.nn.Conv2d(64, 64, 3, 1, 0, bias=False), torch.nn.BatchNorm2d(64)]
        self.features = torch.nn.Sequential(*self.features, *self.features_1, *self.features_2, *self.features_3, *self.features_4, *self.features_5)
        self.extra = torch.nn.ReLU()
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
