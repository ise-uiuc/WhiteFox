
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(12, 3, 1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.permute(v1, 0, 2, 3, 1)
        v3 = v2 - 299
        v4 = F.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 12, 10, 10)
