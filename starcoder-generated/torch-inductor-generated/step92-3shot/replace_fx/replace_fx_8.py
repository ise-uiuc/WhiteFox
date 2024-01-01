
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.b = self.get_params()
    def get_params(self):
        b = torch.nn.Parameter(torch.randn(2))
        return b
    def forward(self, x):
        return x * self.linear.weight + self.b
# Inputs to the model
x = torch.randn(1, 2, 2)
