
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(16, 32, 3, stride=1, padding=1), torch.nn.Conv2d(32, 32, 3, stride=1, padding=1), torch.nn.Conv2d(32, 32, 3, stride=1, padding=1), torch.nn.Conv2d(32, 32, 3, stride=1, padding=1), torch.nn.Conv2d(32, 32, 3, stride=1, padding=1))
        self.lin = torch.nn.Linear(32, 2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1[:, -1, :, :]
        v3 = v2 - 0.2
        v4 = F.relu(v3)
        v5 = self.lin(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 16, 16, 16)
