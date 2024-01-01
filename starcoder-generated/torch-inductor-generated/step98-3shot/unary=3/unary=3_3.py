
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(5, 5, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 * 0.4549257156468935
        v3 = v1 * 0.7630938683971385
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(x1)
        torch.unsqueeze(v7, 0)
        v8 = torch.repeat(v7, 5, 1)
        v9 = v6 * v8
        return v9
# Inputs to the model
x1 = torch.randn(1, 5, 5, 5)
