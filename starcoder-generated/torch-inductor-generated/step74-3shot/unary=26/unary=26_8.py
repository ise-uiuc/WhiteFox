
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(391, 6, 5, stride=(1, 1), bias=False, padding=(2, 2))
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x1 = self.conv_t(x)
        x2 = x1 > 0
        x3 = x1 * 0.012
        x4 = torch.where(x2, x1, x3)
        x5 = self.relu(x4)
        return x5
# Inputs to the model
x = torch.randn(10, 391, 1, 31)
