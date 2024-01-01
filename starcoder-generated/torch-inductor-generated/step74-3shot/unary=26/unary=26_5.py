
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1, bias=False)
        self.conv1 = torch.nn.Conv2d(16, 25, 1, stride=1, padding=0, bias=False)
    def forward(self, x):
        x1 = self.conv_t(x)
        x2 = self.conv1(x1)
        x3 = x2 > 1
        x4 = x2 * 0.61
        x5 = torch.where(x3, x2, x4)
        return torch.nn.functional.relu(x5)
# Inputs to the model
x = torch.randn(6, 16, 24, 18)
