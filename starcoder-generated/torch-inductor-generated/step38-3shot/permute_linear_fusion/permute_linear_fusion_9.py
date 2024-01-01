
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
        self.linear1 = torch.nn.Linear(4, 4)
    def forward(self, x1):
        v5 = torch.randn_like(x1)
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.linear(v2, self.linear1.weight, self.linear1.bias)
        x2 = x2 * v5
        x2 = torch.nn.functional.relu(x2)
        return x2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
