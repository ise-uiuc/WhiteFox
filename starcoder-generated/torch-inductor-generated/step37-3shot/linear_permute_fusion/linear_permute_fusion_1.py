
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = x1
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v2 = v2.permute(0, 2, 1)
        v1 = v1 + 1.0
        return torch.nn.functional.linear(v1, v2, torch.ones([3, 3]))
# Inputs to the model
x1 = torch.randn(1, 1, 1)
