
class Model(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = [
            torch.nn.Conv2d(7, 6, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU(inplace=False),
            torch.nn.ReLU(inplace=False),
            torch.nn.ReLU(inplace=False),
            torch.nn.MaxPool2d(3),
            torch.nn.Conv2d(6, 5, 3, 1, 0)
        ]
    def forward(self, x):
        v = x
        for layer in self.layers:
            v = layer(v)
        return v
# Inputs to the model
x = torch.randn(1, 7, 64, 64)
