
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.relu(v1)
        x2 = torch.nn.functional.linear(v2, self.linear.weight, self.linear.bias)
        x2 = x2.sum(dim=-2) / x2.shape[-2]
        x2 = x2.permute(0, 2, 1)
        v2 = x2.permute(0, 2, 1)
        v3 = v1.shape[1]
        v4 = v1.shape[-2] // v3
        v5 = torch.nn.functional.relu(v2)
        v5 = v5.reshape(1, v4, v3, v5.shape[2])
        v5 = torch.max(v5, dim=-2)[0]
        v5 = torch.max(v5, dim=-2)[0]
        v5 = v5[:, 0, :, 0]
        v3 = v2.shape[-1]
        v2 = v2.reshape(1, -1, v3)
        v2 = (v2 == v3).to(v2.dtype)
        v5 = v5.unsqueeze(dim=-1)
        v3 = v1.shape[-1]
        v4 = (v2 * v5).reshape(v1.shape[1], v1.shape[2], v1.shape[3])
        v4 = torch.sum(v4, -2)
        v4 = v4 / v1.shape[1]
        v4 = v4.permute(1, 0, 2)
        v1 = v1 + v4.to(v1.dtype)
        x1 = x1 * 2
        x1 = torch.sign(x1)
        x1 = torch.tensor(1, dtype=torch.int64).to(x1.dtype)
        v4 = torch.nn.functional.relu(x1)
        return torch.nn.functional.relu(v3)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
