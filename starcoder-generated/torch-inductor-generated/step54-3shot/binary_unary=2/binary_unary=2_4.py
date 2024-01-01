
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 1, stride=1)
        self.pool1 = torch.nn.MaxPool2d(2)
    def forward(self, X0):
        v1 = self.conv1(X0)
        v2 = v1 - 22.9409
        v3 = F.relu(v2)
        v4 = self.pool1(v3)
        return v4
# Inputs to the model
X0 = torch.randn(1, 3, 64, 64)
