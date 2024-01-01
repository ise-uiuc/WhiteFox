
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.relu(v1)
        v2 = x2.detach()
        v3 = v2.unsqueeze(dim=-1)
        return v2 + v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
