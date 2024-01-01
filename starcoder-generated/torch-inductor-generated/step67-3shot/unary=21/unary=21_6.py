
class ModelTanhRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_0 = torch.nn.Conv2d(in_channels=3, out_channels=7, kernel_size=1, bias=False)
        self.tanh_0 = torch.nn.Tanh()
        self.relu_0 = torch.nn.ReLU()
    def forward(self, x0):
        x1 = self.conv_0(x0)
        x2 = self.tanh_0(x1)
        x3 = self.relu_0(x2)
        return x3
# Inputs to the model
x0 = torch.randn(1, 3, 16, 32)
