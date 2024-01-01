
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Embedding(10, 10)
    def forward(self, x1):
        v1 = x1.permute(1, 0)
        v2 = self.linear.permute(1, 0)(v1)
        v3 = x1.permute(0, 2, 1)
        v4 = torch.nn.functional.linear(v3, x1)
        v5 = v2.permute(0, 2, 1) + v4
        v5 = v5.unsqueeze(dim=-2)
        v6 = v5.permute(1, 0, 2)
        v7 = v6.to(v1.dtype)
        return v7
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.arange(10).to(torch.float)
