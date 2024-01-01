
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 5, stride=1, padding=0)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = v1 - x2
        v3 = F.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.tensor([0]).float()
