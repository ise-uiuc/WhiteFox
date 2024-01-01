
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v0 = x1
        v1 = F.linear(v0, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        result = []
        for _ in range(3):
            result.append(v1.permute(0, 2, 1))
        return result[0]
# Inputs to the model
x1 = torch.randn(1, 2, 2)
