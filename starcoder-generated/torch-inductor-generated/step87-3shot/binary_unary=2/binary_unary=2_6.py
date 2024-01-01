
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 4, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = F.relu(torch.sigmoid(v1 - 2))
        v3 = self.conv2(v2)
        v4 = torch.tanh(v3).clamp(-0.5, 3) * 2
        v5 = F.softmax(v4, dim=1)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
