
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.sigmoid = torch.nn.Identity()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = self.sigmoid(v2)
        v4 = v3.squeeze(-1)
        v5 = v4.transpose(1, 2)
        v6 = torch.sum(v5, dim=1, keepdim=True)
        v7 = torch.nn.functional.linear(v6, self.linear.weight, self.linear.bias)
        v8 = self.sigmoid(v7)
        v9 = v8.squeeze(-1)
        v10 = v9.transpose(1, 2)
        v11 = torch.sum(v10, dim=1, keepdim=True)
        return v11
# Inputs to the model
x1 = torch.randn(1, 2, 2)
