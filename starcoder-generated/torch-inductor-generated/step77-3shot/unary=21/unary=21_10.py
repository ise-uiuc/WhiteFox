
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 8,
                                        kernel_size=3, stride=2, padding=1)
    def forward(self, x1):
        x2 = self.conv1(x1)
        x3 = torch.tanh(x2)
        x4 = torch.tanh(x3)
        x5 = torch.tanh(x4)
        return x5
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
