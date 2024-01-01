
class Block1(torch.nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channel_size, channel_size, 1, 1, 0, bias=False)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        v2 = torch.nn.functional.interpolate(concatenated_tensor, size=(802, 971), mode='nearest')
        return v2
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.features_1 = torch.ops.quantized.conv2d(32, 32, (801, 970), stride=[1, 1], padding=[0, 0], bias=None, groups=32)
        self.features_2 = torch.nn.Conv2d(32, 32, 5, 1, 2, bias=False)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        v2 = self.features(concatenated_tensor)
        v3 = (None, v2)
        return (concatenated_tensor, v3)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = torch.nn.Sequential(Model1())
        self.extra= Model1().feature
        self.features = [torch.nn.Conv2d(3, 32, 3, 1, 1, bias=False)]
        self.features_1 = [torch.nn.BatchNorm2d(32)]
        self.features_2 = [torch.nn.ReLU()]
        self.features_3 = [Block1(32)]
        self.features_5 = [torch.nn.Conv2d(64, 64, 3, 1, 0, bias=False), torch.nn.BatchNorm2d(64)]
        self.features = torch.nn.Sequential(*self.features, *self.features_1, *self.features_2, *self.features_3, *self.features_4, *self.features_5)
        self.extra = torch.nn.ReLU()
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
