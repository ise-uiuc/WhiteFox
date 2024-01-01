
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        m1 = torch.nn.ConvTranspose2d(19, 11, 3, stride=2, padding=1, dilation=2, output_padding=2)
        m2 = torch.nn.ConvTranspose2d(11, 19, 3, stride=2, padding=1, dilation=2, output_padding=2)
        self.b = torch.nn.BatchNorm2d(19)
        self.m = torch.nn.ModuleList([m1, m2])
    def forward(self, x):
        for i in range(1):
            x = self.m[i](x)
        return self.b(x)
# Inputs to the model
x = torch.randn(1, 19, 4, 4)
