
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(3, 3, 2, stride=2)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.tanh(v1)
        return self.conv2(v2)


# Inputs to the model
tensor = torch.randn(1, 3, 128, 128)
