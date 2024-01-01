
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 4)
        self.linear1 = torch.nn.Linear(4, 2)
    def forward(self, x1):
        v1 = x1.unsqueeze(dim=-1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v2 = torch.nn.functional.relu(v2)
        v2 = v2.squeeze(dim=-2)
        v3 = torch.max(v2, dim=-1)
        v3 = v3[0]
        return torch.nn.functional.linear(v3, self.linear1.weight, self.linear1.bias)
# Inputs to the model
x1 = torch.randn(1, 1)
