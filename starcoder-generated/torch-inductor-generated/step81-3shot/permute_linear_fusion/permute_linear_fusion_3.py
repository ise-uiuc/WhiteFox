
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 3)
    def forward(self, x1):
        l1 = x1.reshape(x1.size(0), x1.size(1), -1)
        l1 = l1.reshape(-1, x1.size(2))
        v1 = x1.reshape(-1, x1.size(2))
        v4 = v1.permute(0, 2, 1)
        v1 = l1.reshape(-1, v4.size(-1))
        v7 = v1.size(0) // 8
        v5 = v1.reshape(8, -1)
        v8 = v5.unsqueeze(dim=0)
        v5 = v5.unsqueeze(dim=-1)
        v3 = (v5 - v8) ** 2
        v7 = torch.nn.functional.linear(v3, self.linear.weight, self.linear.bias)
        v1 = v7.reshape(v7.size(1), v7.size(2))
        v1 = v1.unsqueeze(dim=0).unsqueeze(dim=-1)
        v9 = v1.reshape(v1.size(0), v1.size(1), v1.size(2), v1.size(3), v1.size(4), -1)
        v1 = v9.reshape(-1, v1.size(1), v1.size(2), v1.size(3), v1.size(4))
        v1 = v1.reshape(v1.size(0), v1.size(1), -1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 5, 5)
