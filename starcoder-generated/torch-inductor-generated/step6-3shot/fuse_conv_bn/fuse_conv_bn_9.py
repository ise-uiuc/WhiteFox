
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_bn = torch.nn.Conv2d(1, 1, 3)
        self.relu = torch.nn.ReLU6()
    def forward(self, x):
        return self.relu(self.conv_bn(x))
# Inputs to the model
x = torch.randn(1, 1, 4, 4)
