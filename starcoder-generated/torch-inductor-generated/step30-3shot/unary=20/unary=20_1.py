
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a1 = torch.nn.ConvTranspose2d(1, 2, 11, 1, 1)
        self.a2 = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.a1(x1)
        v2 = self.a2(v1)
        return v2
# Input to the model
x = torch.randn(1, 1, 44, 44)
