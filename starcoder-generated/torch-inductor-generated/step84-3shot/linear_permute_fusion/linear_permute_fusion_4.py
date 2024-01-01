
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        t1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        t2 = t1.transpose(-1, -2)
        v = torch.nn.functional.relu(t2)
        return v
# Inputs to the model
x1 = torch.randn(1, 2, 2)
