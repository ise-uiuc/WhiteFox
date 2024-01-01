
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.transpose(0, 1)
        v2 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        x2 = torch.nn.functional.linear(v2, self.linear2.weight, self.linear2.bias)
        x3 = torch.nn.functional.sigmoid(x2)
        v3 = x3.detach()
        v4 = torch.max(v3, dim=1)[1]
        v5 = x1.detach()
        v5 = v5.transpose(0, 1)
        v5 = 2*v5
        v5 = v5[v4]
        v3 = x3 + v5.to(dtype=x3.dtype)
        return torch.nn.functional.relu(v3)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
