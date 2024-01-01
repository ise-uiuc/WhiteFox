
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.ConvTranspose2d(3, 5, kernel_size=[3,3], padding=(1,1), bias=False, stride=(3,3))
    def forward(self, x):
        x1 = self.conv(x)
        x2 = x1 > 0
        x3 = x1 * 1.94544
        x4 = torch.where(x2, x1, x3)
        return x4
# Inputs
x = torch.tensor(np.random.rand(3, 3, 15, 22), requires_grad=True)
