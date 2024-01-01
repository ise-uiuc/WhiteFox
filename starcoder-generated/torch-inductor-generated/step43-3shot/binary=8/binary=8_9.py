
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.down = torch.nn.Conv2d(3, 8, 4, stride=2, padding=1)
        self.up = torch.nn.ConvTranspose2d(8, 8, 4, stride=2, padding=1)
        self.act = torch.nn.Sigmoid()
    def forward(self, x1, x2):
        v1 = self.down(x1)
        v2 = self.up(v1)
        v3 = self.act(v2 - x2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
x2 = torch.randn(1, 3, 32, 32)
