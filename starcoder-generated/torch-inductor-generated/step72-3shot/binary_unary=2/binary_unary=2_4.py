
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.Conv2d(4, 2, 10, stride=5, padding=5)
    def forward(self, x1):
        v1 = torch.transpose(x1, 1, 2)
        v2 = self.conv2(v1)
        v3 = v2 - 0.5
        v4 = F.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 4, 16, 16)
