
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = torch.tanh
        self.stack = torch.stack
        self.layer = torch.nn.Linear(2, 3)
        self.flatten = torch.flatten
    def forward(self, x):
        x = self.flatten(self.stack((self.tanh(self.stack((self.layer(x), x))))), 1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
