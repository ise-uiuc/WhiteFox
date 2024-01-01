
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
    def forward(self, x1):
        v2 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.relu(v2)
        return x2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
