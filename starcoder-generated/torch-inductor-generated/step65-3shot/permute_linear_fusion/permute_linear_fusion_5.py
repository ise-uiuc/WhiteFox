
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v1 = v1.permute(0, 2, 1)
        v3 = torch.max(v2, dim=-1)[0]
        x2 = v3.unsqueeze(dim=-1)
        v3 = 10 * x2.to(x2.dtype)
        v4 = (x2!= v3).to(x2.dtype)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x3 = torch.max(v2, dim=-1)[0]
        v4 = v4.permute(0, 2, 1)
        v5 = torch.nn.functional.linear(v4, self.linear.weight, self.linear.bias)
        v5 = torch.max(v5, dim=-1)[0]
        x4 = v5.unsqueeze(dim=-1)
        v5 = 10 * x4.to(x4.dtype)
        v6 = (x4!= v5).to(x4.dtype)
        v6 = torch.max(v6, dim=-1)[0]
        x5 = v6.unsqueeze(dim=-1)
        v6 = 10 * x5.to(x5.dtype)
        v7 = (x5 == 10).to(x5.dtype)
        v7 = v7.permute(0, 2, 1)
        v8 = torch.nn.functional.linear(v7, self.linear.weight, self.linear.bias)
        v8 = torch.max(v8, dim=-1)[0]
        x6 = v8.unsqueeze(dim=-1)
        v8 = 10 * x6.to(x6.dtype)
        v9 = (x6!= v8).to(x6.dtype)
        v9 = (v9 == 1).to(v9.dtype)
        v9 = v9.permute(0, 2, 1)
        v10 = torch.nn.functional.linear(v9, self.linear.weight, self.linear.bias)
        v10 = torch.max(v10, dim=-1)[0]
        return v10
# Inputs to the model
x1 = torch.randn(1, 3, 3)
