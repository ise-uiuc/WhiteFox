
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3, stride=2, padding=1) + torch.nn.ReLU(), torch.nn.Conv2d(3, 3, 3, stride=1, padding=1) + torch.nn.Sigmoid())
    def forward(self, x):
        x1 = self.block(x)
        return x1
# Inputs to the model
x = torch.randn(1, 3, 24, 24)
