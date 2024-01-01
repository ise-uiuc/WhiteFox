
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(8, 3, 1, stride=1, padding=1)
        self.conv2d = torch.nn.Conv2d(8, 3, 1, stride=1, padding=1)
        self.conv3d = torch.nn.Conv3d(8, 3, 1, stride=1, padding=1)
        self.conv4d = torch.nn.Conv4d(8, 3, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1d(x)
        v2 = self.conv2d(x)
        v3 = self.conv3d(x)
        v4 = self.conv4d(x)
        return v1, v2, v3, v4
# Inputs to the model
x = torch.randn(1, 8, 8, 8, 8)
