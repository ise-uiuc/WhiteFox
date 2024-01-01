
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        y = torch.tanh(x1)
        x2 = torch.nn.functional.relu(y)
        y = x2.permute(0, 2, 1)
        x2 = torch.nn.functional.linear(y, self.linear.weight, self.linear.bias)
        return x2
# Inputs to the model
x1 = torch.randn(2, 3, 2)
