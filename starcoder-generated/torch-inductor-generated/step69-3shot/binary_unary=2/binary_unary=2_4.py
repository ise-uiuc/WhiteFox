
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 5, stride=3, padding=2)
    def forward(self, x2):
        v1 = self.conv(x2)
        v2 = torch.transpose(v1, 3, 2)
        v3 = torch.narrow(v2, 2, 7, 8)
        v4 = v3 * 2
        v5 = torch.sum(v4, 3)
        v6 = F.relu(v5)
        return v6
# Inputs to the model
x2 = torch.randn(1, 3, 64, 64)
