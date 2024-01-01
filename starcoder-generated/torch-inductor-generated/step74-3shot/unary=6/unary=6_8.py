
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.transpose(2, 3) + 3
        v3 = v2.unsqueeze(-1)
        v4 = v3.unfold(dimension=2, size=5, step=3)
        v5 = v4.squeeze(-1)
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
