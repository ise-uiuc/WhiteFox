
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 1)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        v3 = torch.nn.functional.relu(v2)
        v4 = torch.max(v3, dim=-1)[0]
        v5 = v4.unsqueeze(dim=-1)
        v5 = v5 + v5
        v6 = (v3 == -1).to(v3.dtype)
        v7 = torch.nn.functional.linear(v6, self.linear2.weight, self.linear2.bias)
        v8 = torch.sum(torch.nn.functional.hardtanh(torch.nn.functional.tanh(v7), -1.0, 1.0))
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
