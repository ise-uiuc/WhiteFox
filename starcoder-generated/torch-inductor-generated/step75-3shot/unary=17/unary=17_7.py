
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convT = nn.ConvTranspose2d(32, 64, 1, stride=1)
        self.convT1 = nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1)
        self.convT2 = nn.ConvTranspose2d(64, 32, 2, stride=2, padding=0)
        self.convT3 = nn.ConvTranspose2d(32, 1, 4, stride=2)
    def forward(self, x1):
        v1 = self.convT(x1)
        v2 = F.relu(v1)
        v3 = self.convT1(v2)
        v4 = F.relu(v3)
        v5 = self.convT2(v4)
        v6 = F.relu(v5)
        v7 = self.convT3(v6)
        v8 = torch.sigmoid(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 32, 100, 100)
