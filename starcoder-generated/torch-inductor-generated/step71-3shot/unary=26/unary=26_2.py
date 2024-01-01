
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose3d(717, 1296, (5, 5, 7), stride=(1, 4, 4), padding=(2, 2, 3), bias=True)
    def forward(self, x100):
        f1 = self.conv_t(x100)
        f2 = f1 > 0
        f3 = f1 * -0.811
        f4 = torch.where(f2, f1, f3)
        return torch.mean(f4)
# Inputs to the model
x100 = torch.randn(1, 717, 14, 11, 23)
