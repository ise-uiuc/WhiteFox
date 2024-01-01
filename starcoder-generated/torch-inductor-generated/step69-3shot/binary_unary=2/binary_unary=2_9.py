
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = torch.unsqueeze(x, 0)
        v2 = self.conv(v1)
        v3 = torch.neg(v2)
        v4 = F.relu(v3)
        v6 = F.relu6(v4)
        v5 = v6[0]
        return v5
# Inputs to the model
x = torch.randn(3, 3, 21, 21)
