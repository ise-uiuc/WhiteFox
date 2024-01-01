
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.ConvTranspose2d(3, 16, 3, stride=1, padding=1)
        self.c2 = torch.nn.ConvTranspose2d(16, 32, 5, stride=2, padding=2)
        self.c3 = torch.nn.ConvTranspose2d(32, 64, 3, stride=2, padding=2)
    def forward(self, x):
        v1 = self.c1(x)
        v2 = self.c2(v1)
        v3 = torch.sigmoid(self.c3(v2))
        return v3
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
