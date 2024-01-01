
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 16, 3, stride=1, padding=1),
            torch.nn.ReLU(),
        )
    def forward(self, x):
        return self.layer1(x)
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
