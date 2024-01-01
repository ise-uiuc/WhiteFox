
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.max(x1.permute(0, 2, 1), dim=-1)[1]
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = v2.T.unsqueeze(-3)
        x2 = v3 + v1
        v4 = torch.nn.functional.relu(x2)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
