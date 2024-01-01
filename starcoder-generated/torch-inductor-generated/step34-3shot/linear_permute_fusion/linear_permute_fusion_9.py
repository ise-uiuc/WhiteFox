
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.cos(x1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = v2.permute(0, 2, 1)
        a1 = v3.unsqueeze(-2)
        v4 = torch.nn.functional.relu(a1)
        return (v2, v4)
# Inputs to the model
x1 = torch.randn(3, 2, 2)
