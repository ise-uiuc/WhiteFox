
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(68, 96, 4, stride=4, padding=0)
    def forward(self, x):
        x1 = self.conv_t(x)
        x2 = x1 > 0
        x3 = x1 * -0.7962
        x4 = torch.where(x2, x1, x3)
        return torch.nn.functional.relu(x4)
# Inputs to the model
x = torch.randn(12, 68, 32, 12)
