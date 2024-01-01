
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 5, stride=2, padding=5)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = -1
        v3 = v1 + v2
        v4 = F.relu(v3) # Or (F.elu(v3) + v2)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
