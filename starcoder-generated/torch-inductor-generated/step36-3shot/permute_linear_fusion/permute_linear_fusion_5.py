
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        x2 = torch.nn.functional.relu(v2)
        v3 = torch.max(x2, dim=-1)[0]
        v4 = v3.unsqueeze(dim=-1)
        v3 = v3 + v4.to(v3.dtype)
        v4 = (v3 == -1).to(v3.dtype)
        v5 = v3.permute(0, 2, 1)
        v3 = torch.nn.functional.linear(v5, self.linear2.weight, self.linear2.bias)
        v3 = torch.max(v3, -1)[0]
        v4 = v3.unsqueeze(dim=-1)
        v3 = v3 + v4.to(v3.dtype)
        v4 = (v3 == -1).to(v3.dtype)
        v3 = torch.max(v3, -1)[0]
        v4 = v3.unsqueeze(dim=-1)
        v3 = v3 + v4.to(v3.dtype)
        v3 = torch.sum(torch.nn.functional.hardtanh(torch.nn.functional.tanh(v3), min_val=-1, max_val=1))
        v3 = torch.reshape(v3, (-1, 1))
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
