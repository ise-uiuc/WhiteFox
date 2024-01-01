
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 3, stride=2, padding=2) # 2nd convolutional layer
        self.conv2 = torch.nn.Conv2d(6, 2, 3, stride=1, padding=1) # 2nd convolutional layer
    def forward(self, x):
        v1 = self.conv2(self.conv1(x))
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 1, 5, 5)
