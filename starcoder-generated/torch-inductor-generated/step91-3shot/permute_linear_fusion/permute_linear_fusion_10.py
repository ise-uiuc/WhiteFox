
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(6, 2)
        self.linear2 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.transpose(1, 2)
        v2 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        v3 = torch.nn.functional.linear(self.linear2.weight.to(v1.dtype) * v2, self.linear2.weight, self.linear2.bias)
        v4 = v3.squeeze(-1)
        v5 = v4.transpose(1, 2)
        return torch.sum(v5, dim=1, keepdim=True)
# Inputs to the model
x1 = torch.randn(1, 2, 6)
