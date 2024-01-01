
class Model(nn.Module):
    def forward(self, x1):
        v1 = nn.ConvTranspose1d(5, 2, kernel_size=3)
        v2 = nn.ConvTranspose2d(5, 2, kernel_size=3)
        v3 = nn.ConvTranspose3d(5, 2, kernel_size=3)
        v4 = v3.forward(x1)
        v5 = v2.forward(v4)
        v6 = v1.forward(v5)
        return v6
# Inputs to the model
x1 = torch.randn(3, 5, 3)
