
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
    def forward(self, x1):
        v5 = torch.randn_like(x1)
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v2 = v2 + v2
        x2 = torch.nn.functional.relu(v2)
        v3 = x2.detach()
        v4 = torch.max(v3, dim=-1)[1]
        v4 = v4.unsqueeze(dim=-1)
        v3 = v3 + v4.to(v3.dtype)
        v4 = (v3 == -1).to(v3.dtype)
        return v4 * v5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
