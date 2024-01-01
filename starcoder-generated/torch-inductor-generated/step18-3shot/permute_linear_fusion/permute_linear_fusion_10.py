
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v2 = torch.cat([v2, v2, v2], dim=2)
        v2 = v2 - v1
        x2 = torch.nn.functional.relu(v2)
        v3 = -1.0 * torch.abs(x2)
        x2 = x2 - v3
        v3 = v2.size()
        v4 = v3[0]
        v4 = v4 * 2
        v4 = v4 + 2
        v4 = v4 + 1
        v5 = v4 * v3[3]
        v5 = v5.unsqueeze(dim=-1)
        v5 = v5.expand_as(x2)
        v6 = -32 * v5
        v5 = v5.expand_as(x2)
        v6 = v6 + v5
        v6 = 0.8 * v6
        v4 = v4 + 1
        v2 = v2 - v6
        v6 = torch.sign(v6)
        v6 = -1 * v6
        v6 = torch.clamp(v6, max=1)
        v6 = torch.nn.functional.max_pool2d(v6, 7, 1, 3)
        v2 = v2 * v6
        v5 = v1 + v2
        v3 = v2.expand_as(v1)
        v4 = v4.expand_as(v1)
        v7 = v4.expand_as(v1)
        v5 = v5 * v7
        v2 = v1 < v7
        v8 = v2.to(v1.dtype)
        v5 = v5 + v8
        v8 = torch.nn.functional.max_pool1d(v8, 1)
        v5 = v5 + v8
        v4 = torch.nn.functional.max_pool2d(v4, 1)
        v6 = torch.nn.functional.tanh(v1)
        v5 = v5 + v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
