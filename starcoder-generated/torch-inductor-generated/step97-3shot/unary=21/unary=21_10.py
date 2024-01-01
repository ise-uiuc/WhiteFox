
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 =torch.nn.Conv2d(9, 19, 1, stride=1)
        self.conv2 =torch.nn.Conv2d(19, 29, 1, stride=1)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = torch.tanh(x1)
        return self.conv2(x2)
# Inputs to the model
x = torch.randn(1, 9, 1, 1)
