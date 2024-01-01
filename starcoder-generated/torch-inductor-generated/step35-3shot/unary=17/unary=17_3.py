
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(15, 16, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = torch.sigmoid(v2)
        v4 = torch.tanh(v3)
        v5 = (torch.transpose(v4, 2, 1))
        return v5
# Inputs to the model
x1 = torch.randn(1, 15, 24, 12)
