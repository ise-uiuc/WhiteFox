
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
    def forward(self, x):
        v4 = self.linear(x)
        v5 = v4.shape
        v0 = self.linear
        return v5
# Inputs to the model
x = torch.randn(2, 4)
