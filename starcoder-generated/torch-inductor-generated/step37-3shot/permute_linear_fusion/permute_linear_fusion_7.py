
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        v3 = torch.nn.functional.linear(v2, self.linear2.weight, self.linear2.bias)
        x2 = torch.relu(v3)
        v4 = x2.detach()
        v5 = torch.max(v4, dim=-1)[1]
        v5 = v5.unsqueeze(dim=-1)
        v4 = v4 + v5.to(v4.dtype)
        v5 = (v4 == -1).to(v4.dtype)
        x3 = (v4.T + v5 + v2).T
        x3 = x3.permute(0, 2, 1)
        x4 = torch.nn.functional.linear(x3, self.linear2.weight, self.linear2.bias)
        return torch.relu(x4)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
