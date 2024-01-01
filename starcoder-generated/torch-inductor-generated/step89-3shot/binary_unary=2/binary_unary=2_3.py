
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, 5, stride=1, padding=1)
    def forward(self, x1):
        x2 = self.conv1(x1)
        x3 = x2 - 1
        x4 = F.relu(x3)
        x5 = torch.squeeze(x4, 0)
        return x5
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
