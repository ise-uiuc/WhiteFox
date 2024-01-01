
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv_2 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.Conv_3 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.Conv_1 = nn.Conv2d(16, 1, 3, stride=1, padding=1)
    def forward(self, x_t):
        v1 = self.Conv_2(x_t)
        v2 = self.Conv_3(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 32, 64)
