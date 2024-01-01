
class Block1(torch.nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channel_size, channel_size, 1, 1, 0, bias=False)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        v2 = torch.nn.functional.interpolate(concatenated_tensor, size=(802, 971), mode='nearest')
        return v2
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.features_1 = Block1(32)
        self.features_2 = torch.nn.Conv2d(32, 32, 5, 1, 2, bias=False)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        v2 = self.features(concatenated_tensor)
        v3 = (None, v2)
        return (concatenated_tensor, v3)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
