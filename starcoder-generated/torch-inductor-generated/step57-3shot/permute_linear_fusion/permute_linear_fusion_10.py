
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
        self.linear2 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x1 = torch.nn.functional.relu(v2)
        v3 = torch.nn.functional.linear(x1, self.linear2.weight, self.linear2.bias)
        return v3
# Inputs to the model
x1 = torch.randn(1, 4, 2)
