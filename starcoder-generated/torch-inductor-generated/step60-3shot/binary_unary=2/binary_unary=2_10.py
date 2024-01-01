
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 1, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 0.321
        v3 = F.elu(v2, alpha=0.1)
        v4 = F.softmax(v3, dim=0)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)
