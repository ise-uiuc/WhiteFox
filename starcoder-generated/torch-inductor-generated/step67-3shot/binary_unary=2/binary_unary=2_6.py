
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 64, 5, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.Tensor.shape(v1)[-1]
        v3 = torch.shape(v2)
        v4 = v3[-1]
        v5 = 0 + v4
        v6 = v5 - 0.5
        v7 = F.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 64, 32, 32)
