
class Block1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 32, 1, 1, 0, bias=False)
        self.conv2 = torch.nn.ConvTranspose2d(32, 32, 2, 2, 0, bias=False)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        v2 = self.conv1(concatenated_tensor)
        v2 = torch.nn.functional.interpolate(v2,size=(802, 971), mode='bilinear', align_corners=False)
        v2 = self.conv2(v2)
        return torch.nn.functional.pad(v2, [3,3,3,3])
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.features_1 = Block1()
        self.features_2 = torch.nn.Conv2d(32, 32, 5, 1, 2, bias=False)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        v2 = self.features(concatenated_tensor)
        v3 = self.features_1(v2)
        v4 = self.features_2(v3)
        return (concatenated_tensor, torch.split(v4, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
