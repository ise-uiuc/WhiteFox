
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(4096, 2048)
        self.linear2 = torch.nn.Linear(2048, 1024)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        x2 = torch.nn.functional.softplus(v2)
        x2 = self.linear2(x2)
        x2 = torch.nn.functional.gelu(x2, approximate=False)
        x2 = torch.nn.functional.gelu(x2, approximate=False)
        return torch.max(x2, dim=-1)[0]
# Inputs to the model
x1 = torch.randn(1, 4096, 2)
