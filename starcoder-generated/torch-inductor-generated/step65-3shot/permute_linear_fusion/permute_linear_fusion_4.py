
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = torch.max(v1, dim=-1)[0]
        v2 = v2.unsqueeze(dim=-1)
        v3 = v2.permute(0, 2, 1)
        v4 = torch.nn.functional.linear(v3, self.linear.weight, self.linear.bias)
        v5 = torch.max(v4, dim=-1)[0]
        v6 = v5.unsqueeze(dim=-1)
        v5 = v5 + v6.to(v5.dtype)
        v6 = (v5 == -1).to(v5.dtype)
        v5 = torch.max(v5, dim=-1)[0]
        v5 = v5.unsqueeze(dim=-1)
        v5 = v5 + v6.to(v5.dtype)
        v4 = torch.reshape(v5, (-1, 1))
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
