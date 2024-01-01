
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=1, stride=2)
        self.conv2 = torch.nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=1, output_padding=1)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        return out
# Inputs to the model
x = torch.randn(1, 3, 5, 5)
