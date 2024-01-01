
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.linear1 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.detach()
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.relu(v2)
        v3 = x2.detach()
        v4 = torch.nn.functional.linear(v3, self.linear1.weight, self.linear1.weight)
        v5 = v4.unsqueeze(dim=-1)
        v6 = torch.randint_like(v4, low=-1, high=1)
        x2 = torch.randint(low=-1, high=0, size=(1, 2, 22), dtype=torch.int64, layout=torch.strided, device=None, requires_grad=False)
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
