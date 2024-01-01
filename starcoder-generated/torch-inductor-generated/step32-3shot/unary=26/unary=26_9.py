
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(4, 128, 7, stride=2, padding=2, bias=False)
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x1):
        x1 = self.conv_t(x1)
        x1 = self.relu(x1)
        return x1
# Inputs to the model
x1 = torch.randn(3, 4, 35, 42)
