
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 128, 5, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(128, 256, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2[:, :, :, :]
        v4 = torch.nn.functional.relu6(v3)
        v5 = v4 - 1.1 
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
