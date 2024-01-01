
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(in_channels=11, out_channels=30, kernel_size=(2,2))
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.tanh(x1)
        return x2
# Inputs to the model
x = torch.randn(1, 11, 59, 59)
