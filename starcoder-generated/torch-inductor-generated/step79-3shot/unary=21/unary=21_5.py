
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=1, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 16, kernel_size=1, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=1)
    def forward(self, x):
        conv1 = self.conv1(x)
        x1 = torch.tanh(conv1)
        conv2 = self.conv2(x1)
        x2 = torch.tanh(conv2)
        conv3 = self.conv3(x2)
        x3 = torch.tanh(conv3)
        return x3
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
