

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
        self.linear3 = torch.nn.Linear(2, 1)
    def forward(self, x1):
        if self.linear3.bias is None:
            raise AssertionError('Bias is not used in linear layer')
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        x2 = torch.nn.functional.relu(v2)
        v3 = torch.nn.functional.linear(x2, self.linear2.weight, self.linear2.bias)
        v4 = v3.detach()
        v5 = v1.permute(0, 2, 1)
        v6 = torch.nn.functional.linear(v5, self.linear3.weight, self.linear3.bias)
        x3 = torch.nn.functional.relu(v6)
        v7 = x3.detach()
        v8 = torch.max(v7, dim=-1)[1]
        v8 = v8.unsqueeze(dim=-1)
        v7 = v7 + v8.to(v7.dtype)
        v8 = (v7 == -1).to(v7.dtype)
        v7 = torch.nn.functional.linear(v7, self.linear3.weight, self.linear3.bias)
        x4 = torch.cat([v2, v7], dim=-1)
        v9 = x4
        v10 = torch.max(v9, dim=-1)[0]
        v11 = v10.unsqueeze(dim=-1)
        v9 = v9 + v11.to(v9.dtype)
        v11 = (v9 == -1).to(v9.dtype)
        v9 = v9 * v11
        v10 = torch.max(v9, dim=-1)[1]
        return v10
# Inputs to the model
x1 = torch.randn(1, 2, 3)
