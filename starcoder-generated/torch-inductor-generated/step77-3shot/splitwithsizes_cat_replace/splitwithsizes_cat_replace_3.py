
class Block1(torch.nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channel_size, channel_size, 1, 1, 0, bias=False)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (torch.split(concatenated_tensor, [3], dim=1), torch.split(concatenated_tensor, [3], dim=1))
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.features_1 = Block1(32)
        self.features_2 = torch.nn.Conv2d(32, 32, 5, 1, 2, bias=False)
    def forward(self, v1):
        v2 = torch.split(v1, [1, 1, 1], dim=1)
        v3 = self.features(torch.cat(v2, dim=1))
        v4 = (None, v3)
        return (None, v4)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
