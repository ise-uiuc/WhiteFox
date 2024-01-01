
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        x2 = self.linear(x1)
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x3 = x2.clone()
        x3 = torch.nn.functional.relu(x3)
        v3 = torch.max(x2.clone(), dim=1)[-1]
        v3 = v3.unsqueeze(-1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
