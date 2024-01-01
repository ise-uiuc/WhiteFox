
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, 3, stride=1, padding=1),
            torch.nn.Conv2d(1, 1, 3, stride=1, padding=1),
            torch.nn.Conv2d(1, 1, 3, stride=1, padding=1),
        )
    def forward(self, x):
        return self.layer1(x)
# Inputs to the model
x = torch.randn(1, 1, 28, 28)
