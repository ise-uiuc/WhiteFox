
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.linear.bias = torch.nn.Parameter(torch.tensor([[False, False, True, False], [False, False, False, False]], dtype=bool), requires_grad=True)
    def forward(self, x1):
        v1 = torch.relu(x1)
        v2 = torch.relu(v1)
        v4 = v2
        v8 = self.linear.weight
        v1 = torch.nn.functional.linear(v4, v8, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        return v2
# Inputs to the model
x1 = torch.randn(2, 5, 2)
