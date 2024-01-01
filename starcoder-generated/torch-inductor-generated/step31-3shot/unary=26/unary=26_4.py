
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.ConvTranspose2d(1, 3, 3, stride=1, padding=0, output_padding=0, groups=1, bias=False)
    def forward(self, x):
        x = self.conv(x)
        x = x > 0
        x = x * -0.009009742276873541
        x = torch.where(x, x, x)
        return x
# Inputs to the model
x = torch.randn(2, 1, 4, 4)
