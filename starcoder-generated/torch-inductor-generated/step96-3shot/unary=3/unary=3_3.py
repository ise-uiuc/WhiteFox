
class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.conv = torch.nn.Conv2d(12, 16, 2, stride=1, padding=0)
        self.conv21 = torch.nn.ConvTranspose2d(16, 64, 3, stride=4, padding=1)
        self.conv22 = torch.nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1)
    def forward(self, X):
        Y1 = self.conv(X)
        Y2 = Y1 * 0.5
        Y3 = Y1 * 0.7071067811865476
        Y4 = torch.erf(Y3)
        Y5 = Y4 + 1
        Y6 = Y2 * Y5
        Y7 = self.conv21(Y6)
        Y8 = Y7 * 0.5
        Y9 = Y7 * 0.7071067811865476
        Y10 = torch.erf(Y9)
        Y11 = Y10 + 1
        Y12 = Y8 * Y11
        Y13 = self.conv22(Y12)
        return Y13
# Inputs to the model
X = torch.randn(1, 12, 28, 28)
