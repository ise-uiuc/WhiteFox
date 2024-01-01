
with torch.no_grad():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)
        def forward(self, x1, x2):
            x2 = (x2 - torch.max(x2, dim=-1)[0]) * torch.tensor([2]) + torch.max(x2, dim=-1)[0]
            x2 = torch.gt(x1, x2).to(x1.dtype) + torch.lt(x2, x1).to(x1.dtype)
            v1 = x1.permute(0, 2, 1)
            v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
            x2 = torch.nn.functional.relu(v2)
            v3 = x2.detach()
            v4 = torch.max(v3, dim=-1)[1]
            v4 = v4.unsqueeze(dim=-1)
            v3 = v3 + v4.to(v3.dtype)
            return torch.nn.functional.linear(v3, self.linear.weight, self.linear.bias)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
