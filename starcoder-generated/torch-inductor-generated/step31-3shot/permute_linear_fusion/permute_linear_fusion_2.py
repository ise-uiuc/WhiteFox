
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.linear = lambda x: torch.nn.functional.linear(x, self.linear.weight, self.linear.bias)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = self.linear(v1)
        v2 = v2.detach()
        v3 = torch.max(v2, dim=-1)[0]
        v4 = torch.max(v2, dim=-1)[1]
        v4 = v4.unsqueeze(dim=0)
        v5 = v2.shape[0]
        v4 = v4.repeat(v5, 1)
        v3 = v3.unsqueeze(dim=-1)
        v3 = v3.repeat(1, 1, 2)
        v3 = v3 + v4
        v3 = v3 > 0
        v4 = v3.to(x1.dtype)
        return (v2 * v4)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
