
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(1, 3, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 1, 65, 63)
