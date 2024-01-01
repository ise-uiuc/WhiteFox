
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.a = nn.ConvTranspose2d(1, 2, 3, stride = 2, output_padding=1, bias=True)
        self.b = nn.ReLU(inplace=True)
        self.c = nn.ConvTranspose2d(2, 1, 3, stride=2, output_padding=0, bias=False)
        self.d = nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True)
    def forward(self, x):
        x = self.a(x)
        x = self.b(x)
        x = self.c(x)
        x = self.d(x)
        return x
# Inputs to the model
x = torch.randn(1,1,10,10)
