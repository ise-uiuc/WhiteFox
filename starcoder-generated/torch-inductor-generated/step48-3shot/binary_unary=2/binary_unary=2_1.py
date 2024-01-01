
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, stride=2, padding=1),
            torch.nn.Conv2d(64, 128, 1, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 64, 1, stride=1, padding=0)
        )
    def forward(self, x1):
        return self.layers(x1)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
