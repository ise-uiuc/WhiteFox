
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 132, 5, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(132, 16, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.transpose(v3, 1, 2)
        v5 = torch.transpose(v4, 1, 2)
        return v5
# Inputs to the model
x1 = torch.randn(1, 16, 56, 56)
