
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 5, stride=1, padding=2)
    def forward(self, x1):
        v1 = F.relu(x1)
        v2 = F.sigmoid(x1)
        v3 = self.conv1(v1)
        v4 = v3 - v2
        v5 = F.tanh(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
