
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = v2.unsqueeze(1)
        v4 = self.linear.weight * v3
        v5 = v2.shape
        v6 = v2.reshape(-1, 2)
        v4 = v4.reshape(v5)
        v7 = v2 >= v4
        v8 = v2 > v4
        v9 = v7 | v8
        v10 = v9.squeeze(1)
        v10 = v9.squeeze(-1)
        return v10
# Inputs to the model
x1 = torch.randn(1, 2, 2)
