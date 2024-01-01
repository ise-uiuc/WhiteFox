
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(16, 1000, bias=True)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.relu(v1)
        v3 = v2.view(-1, 16)
        v4 = self.fc(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
