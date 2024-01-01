
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1).expand_as(self.linear.weight)
        v1 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.relu(v1).clone()
        v2 = torch.max(x2.clone(), dim=1)[-1]
        v2 = v2.unsqueeze(-1)
        return v2
# Inputs to the model
x1 = torch.randn(4, 2, 2)
