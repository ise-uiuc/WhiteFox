
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v1 = v1.permute(0, 2, 1)
        v3 = torch.max(v2, dim=-1)[0]
        v3 = torch.sqrt(torch.sum(torch.abs(v3), dim=1, keepdim=True))
        v4 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.relu(v3)
        return x2, v4
# Inputs to the model
x1 = torch.randn(1, 3, 3)
