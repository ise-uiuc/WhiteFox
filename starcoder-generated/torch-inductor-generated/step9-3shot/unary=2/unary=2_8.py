
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(2, 2, kernel_size=5, stride=1, padding=2)
        self.conv_t2 = torch.nn.ConvTranspose2d(2, 4, kernel_size=3, stride=1, padding=1)
        self.conv_t3 = torch.nn.ConvTranspose2d(4, 2, kernel_size=3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_t1(x1)
        v2 = self.conv_t2(x1)
        v3 = self.conv_t3(v2)
        v4 = v2 * v1
        v5 = v4 * v3
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 3, 3)
