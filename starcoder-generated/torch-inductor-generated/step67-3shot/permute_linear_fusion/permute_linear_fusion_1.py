
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(1, 2, 0)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.relu(v2)
        v3 = x1.permute(1, 0, 2)
        v4 = torch.nn.functional.tanh(v3)
        x2 = torch.sqrt(torch.sum(torch.abs(x2)))
        x3 = x1 + x2
        return (x3, x1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
