
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.randn_like(x1[:, :1, :])
        v1 = v1.to(torch.int32)
        v2 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v2, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.relu(v2)
        v3 = torch.matmul(x1, v1)
        return torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
