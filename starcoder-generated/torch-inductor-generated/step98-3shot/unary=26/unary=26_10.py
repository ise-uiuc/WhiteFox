
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(7, 4, 2, stride=1, padding=1, output_padding=0, bias=False)
        self.flatten = torch.nn.Flatten()
    def forward(self, x3):
        x4 = torch.nn.ReLU()(self.conv_t(x3))
        x5 = self.flatten(x4)
        x6 = torch.nn.LeakyReLU()(torch.nn.Linear(196, 6)(x5))
        return x6.unsqueeze(1)
# Inputs to the model
x3 = torch.randn(5, 7, 8, 13)
