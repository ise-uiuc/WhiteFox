
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 - 1.1633
        return v2
# Inputs to the model
x = torch.randn(1, 2)
