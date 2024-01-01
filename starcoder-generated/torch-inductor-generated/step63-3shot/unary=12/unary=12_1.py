
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.mul = torch.bmm
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.sigmoid(v1)
        v3 = self.mul(v1.unsqueeze(2),v2.unsqueeze(3)).squeeze(-1).squeeze(-1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
