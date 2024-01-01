
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5)
        self.conv2 = torch.nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x.nn.Conv2d(20, 20, 5)
# Inputs to the model
x = torch.randn(5, 5, 5)
