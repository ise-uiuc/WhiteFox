
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
        self.linear3 = torch.nn.Linear(3, 3)
    def forward(self, x1):
        v1 = torch.cat([x1, x1, x1], dim=-1)
        v2 = v1.permute(0, 2, 1)
        v21 = v2
        v22 = torch.nn.functional.relu(self.linear.weight) * v2 + self.linear.bias
        v2 = self.linear3(torch.nn.functional.relu(v22))
        v3 = v2.mean(dim=-2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
