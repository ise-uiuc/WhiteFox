
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = [torch.randn(2, 1) for _ in range(3)]
    def forward(self, x1, x2):
        result = []
        count = 0
        for item in self.weights:
            v1 = torch.mm(x1, self.weights[3 - count])
            v2 = torch.mm(x2, self.weights[7 - count])
            result.append([v1, v2])
            count += 1
        return torch.cat([torch.cat(item, 1) for item in result], 0) # 7 vs 9 in example
# Inputs to the model
x1 = torch.randn(1, 2)
x2 = torch.randn(1, 2)
