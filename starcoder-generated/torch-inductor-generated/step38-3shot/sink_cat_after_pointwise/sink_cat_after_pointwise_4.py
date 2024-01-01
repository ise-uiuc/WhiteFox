
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._weight = torch.nn.Parameter(torch.randn(1, 2 * 3 * 4) + 0.01)
    def forward(self, x):
        return x * self._weight
# Inputs to the model
x = torch.randn(2, 3, 4)
# Input to copy the weight
w = torch.randn(2, 28)
