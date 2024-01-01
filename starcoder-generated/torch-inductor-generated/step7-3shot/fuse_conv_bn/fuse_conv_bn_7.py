
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        m0 = nn.Conv2d(2, 4, 3, stride=(2, 4))
        m1 = nn.BatchNorm2d(4)
        m2 = nn.Conv2d(4, 6, 3, stride=(1, 1))
        m3 = nn.BatchNorm2d(6)
        self.model = nn.Sequential(m0, nn.ReLU(inplace=True), m0)
        self.model2 = nn.Sequential(m1, m2, m3)
    def forward(self, x):
        x = self.model(x)
        return self.model2(x)
# Inputs to the model
x = torch.randn(2, 2, 5, 6)
