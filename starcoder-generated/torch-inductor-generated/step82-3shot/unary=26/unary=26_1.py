
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(10, 12, 7, stride=1, padding=0, output_padding=1, bias=False)
        self.conv = torch.nn.Conv2d(12, 6, 1, padding=0, bias=False)
    def forward(self, x):
        x = torch.transpose(x, 0, 1)
        x1 = self.conv_t(x)
        x2 = self.conv(x1)
        x = torch.flip(x2, dims=[0])
        x = torch.transpose(x, 0, 1)
        x = torch.transpose(x, 0, 2)
        x = torch.flip(x, dims=[0])
        x = torch.flip(x, dims=[0, 1])
        return x
# Inputs to the model
x = torch.rand(24, 10, 6, 10)
