
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(1, 31, 2, 3, padding=1)
        self.conv_t2 = torch.nn.ConvTranspose2d(31, 31, 2, 3, padding=1)
    def forward(self, x):
        v1 = self.conv_t1(x)
        v2 = self.conv_t2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 6, 4)
