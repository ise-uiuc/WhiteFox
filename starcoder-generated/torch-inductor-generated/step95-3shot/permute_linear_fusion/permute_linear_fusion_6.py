
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.tanh(x1)
        v1 = v1.abs()
        v1 = v1.ceil()
        v1 = v1.detach()
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.relu(v2)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
