
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v = []
        indices = torch.randperm(49)
        for i in range(49):
            v.append(torch.mm(x1, x2))
        indices4 = torch.cat([indices[i * 5 : (i + 1) * 5] for i in range(9)], 0)
        indices5 = [indices4[i % 10] + i // 10 * 36 for i in range(49)]
        indices6 = torch.tensor(indices5)
        return torch.cat([v[i] for i in indices6], 1)
# Inputs to the model
x1 = torch.randn(20, 5)
x2 = torch.randn(5, 20)
