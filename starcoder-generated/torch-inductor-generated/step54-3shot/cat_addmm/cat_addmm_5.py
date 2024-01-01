
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = nn.ConvTranspose2d(3, 2, kernel_size=2)
        self.t2 = nn.ConvTranspose2d(2, 3, kernel_size=2)
    def forward(self, x):
        x = self.t1(x)
        x = self.t2(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 3, 3)
