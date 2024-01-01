
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 15)
        self.activation = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(16, 16, 15)
        self.batch_norm = torch.nn.BatchNorm2d(16)
    def forward(self, x):
        x = self.conv1(x)
        x1 = self.activation(x)
        x2 = self.conv2(x1)
        y = self.batch_norm(x2)
        return y
# Inputs to the model
x = torch.randn(4, 16, 900, 1000)
