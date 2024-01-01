
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.linear3 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.relu(v2)
        with torch.no_grad():
            v2 = v2.permute(0, 2, 1)
        v3 = v2 + v1 + self.linear3.weight
        v2 = x2.permute(0, 2, 1)
        v1 = torch.nn.functional.relu(v1)
        return torch.nn.functional.linear(v1, v3, self.linear3.bias.data)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
