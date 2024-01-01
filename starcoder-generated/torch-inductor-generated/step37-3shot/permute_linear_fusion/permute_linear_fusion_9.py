
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.relu(v2)
        v3 = torch.nn.functional.gelu(v2)
        x3 = (v3 + 0.5).floor().to(torch.int64)
        v4 = torch.nn.functional.softmax(x3, -1)
        v4 = torch.max(v4, dim=-1).values
        x4 = v4.unsqueeze(dim=1)
        v3 = torch.mul(v1, x4)
        x5 = v3.floor().to(torch.int64)
        x5 = torch.clamp(x5, 0, 1)
        x6 = v3.floor().to(torch.int64)
        x6 = torch.clamp(x6, -1, 0)
        v5 = x5 - x6
        v6 = v5 * 2
        v7 = torch.nn.functional.softmax(v6, -1)
        v8 = v7.unsqueeze(dim=1)
        return torch.nn.functional.linear(v5, v8)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
