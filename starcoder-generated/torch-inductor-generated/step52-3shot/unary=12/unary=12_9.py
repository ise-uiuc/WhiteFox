
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(7, 6, [3, 3, 3], stride=[2, 1, 1], padding=[0, 1, 1], dilation=3)
        self.conv2 = torch.nn.Conv1d(6, 4, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv2(v1)
        return v2
# Inputs to the model
x = torch.Tensor(1, 7, 10, 10, 10)
