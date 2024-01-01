
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv2d = torch.nn.Conv2d(3, 3, (3, 3))
    def forward(self, x1):
        return self.Conv2d(x1)
# Inputs to the model
x1 = torch.randn(1, 3, 1, 1)
