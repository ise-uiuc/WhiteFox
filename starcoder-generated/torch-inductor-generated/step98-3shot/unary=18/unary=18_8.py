
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, (2, 2), stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 16, (3, 3), stride=1, padding=0)
        self.linear = torch.nn.Linear(289, 10)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        v5 = v4.view(v4.size(0), -1)
        v6 = self.linear(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
