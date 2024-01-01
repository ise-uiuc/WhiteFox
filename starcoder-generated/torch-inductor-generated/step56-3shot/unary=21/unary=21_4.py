
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(5, 4, 3)
        self.conv2 = torch.nn.Conv2d(4, 5, 3)
    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(x)
        v2 = torch.tanh(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(1, 5, 224, 224, requires_grad=False)
