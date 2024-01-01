
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.relu(v2)
        v3 = x2.tanh()
        v4 = torch.tanh(v3)
        v5 = torch.mean(v4, -1)
        v5 = v5.unsqueeze(-1)
        return v3 * v5 + x2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
