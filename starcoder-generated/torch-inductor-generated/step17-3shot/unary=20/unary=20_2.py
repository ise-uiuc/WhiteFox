
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = nn.ConvTranspose2d(4, 5, kernel_size=4, stride=1, padding=1)
        self.conv_transpose2 = nn.ConvTranspose2d(4, 5, kernel_size=4, stride=1, padding=1)
    def forward(self, x):
        t1 = self.conv_transpose2(x)
        t2 = self.conv_transpose1(x)
        return t1, t2
# Inputs to the model
x = torch.randn(1, 4, 8, 8)
