
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 3, 1, 1)
    def forward(self, v1, v2, v3, v4):
        split_tensors = torch.split(self.conv(v1), [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat([self.conv(v2), self.conv(v3), self.conv(v4)], dim=1)
        f = torch.sigmoid(concatenated_tensor + torch.sigmoid(split_tensors[0]) + torch.sigmoid(split_tensors[1]) + torch.sigmoid(split_tensors[2]))
        return (concatenated_tensor, split_tensors, f)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 3, 64, 64)
x4 = torch.randn(1, 3, 64, 64)
