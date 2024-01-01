
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c4 = torch.nn.Conv2d(3, 4, 2)
    def forward(self, x):
        v1 = self.c4(x)
        y = torch.relu(v1)
        return y
# Inputs to the model
x = torch.randn(1, 3, 10, 10)
