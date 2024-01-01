
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.relu(v2)
        v3 = v2.permute(0, 2, 1)
        x2 = x2.permute(0, 2, 1)
        v3 = torch.matmul(x2, v3)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
