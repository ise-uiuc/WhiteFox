
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.Sequential(torch.nn.ConvTranspose2d(104, 80, 6, stride=2, padding=1, bias=False), torch.nn.ReLU(), torch.nn.ConvTranspose2d(80, 16, 4, stride=2, padding=1, bias=False), torch.nn.ReLU(), torch.nn.ConvTranspose2d(16, 2, 4, stride=2, padding=1, bias=False))
    def forward(self, x2):
        r1 = self.conv_t(x2)
        return torch.nn.functional.interpolate(torch.nn.ReLU()(r1), (33, 66))
# Inputs to the model
x2 = torch.randn(2, 104, 66, 33)
