
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.transpose(self.linear.weight, 0, 1)
        v3 = v2.unsqueeze(dim=3)
        x2 = v2.unsqueeze(dim=2)
        v4 = torch.matmul(x2, v3)
        v7 = torch.matmul(v1, v2)
        v6 = v4 + v7
        v8 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        return v8
# Inputs to the model
x1 = torch.randn(1, 2, 2)
