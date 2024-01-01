
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3, 2, 5, stride=1, padding=2, output_padding=1),
            torch.nn.ConvTranspose2d(2, 1, 2, stride=2, padding=1, output_padding=1),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        return self.deconv(x)
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
