
class Example(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 10, 5),
            torch.nn.BatchNorm2d(10),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(10, 2, 5))
    def forward(self, x):
        x = self.layers(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 10, 10)
