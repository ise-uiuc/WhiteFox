
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 1)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        x2 = torch.nn.functional.relu(v2)
        v3 = torch.nn.functional.linear(x2, self.linear2.weight, self.linear2.bias)
        x2 = torch.nn.functional.tanh(v3)
        v3 = torch.max(v2, dim=-1)[0]
        v4 = v3.unsqueeze(dim=-1)
        v3 = v3 + v4.to(v3.dtype)
        v4 = (v3 == -1).to(v3.dtype)
        return torch.nn.functional.linear(v4, self.linear2.weight, self.linear2.bias)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
