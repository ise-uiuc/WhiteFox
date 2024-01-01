
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=1, padding=0, stride=1)
        self.conv2 = torch.nn.Conv2d(64, 32, kernel_size=1, stride=1)
    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(x)
        return self.conv2(x)
# Inputs to the model
x = torch.randn(1,3,1,1)
