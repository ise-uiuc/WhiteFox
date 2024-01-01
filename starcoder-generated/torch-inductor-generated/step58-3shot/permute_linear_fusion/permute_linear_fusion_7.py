
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.relu(v2)
        x3 = x2.detach()
        x4 = x1.min(dim=-1)[1]
        v1 = x3.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        return torch.nn.functional.linear(v2, self.linear.weight, self.linear.bias)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
