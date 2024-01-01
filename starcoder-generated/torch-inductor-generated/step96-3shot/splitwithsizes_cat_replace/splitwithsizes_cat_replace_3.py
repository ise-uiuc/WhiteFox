
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features1 = torch.nn.MaxPool2d(5, 1, 2)
        self.features2 = torch.nn.MaxPool2d(5, 1, 4)
    def forward(self, x):
        x = self.features1(x)
        y = x
        x = self.features2(x)
        x = y.add(x)
        return (x, (x, y))
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
