
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(262, 256, kernel_size=(4, 3), stride=(1, 1), padding=(0, 0), groups=256),
            torch.nn.Sigmoid(),
        )
    def forward(self, x1):
        v1 = self.conv_t(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 262, 35, 35)
