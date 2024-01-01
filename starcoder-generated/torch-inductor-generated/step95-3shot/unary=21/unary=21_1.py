
class ModelLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x):
        return self.linear(x)
# Inputs to the model
x = torch.randn(1, 1, 1, 64)
