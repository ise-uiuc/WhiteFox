
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x1 = nn.Linear(5, 4)
        self.x2 = nn.Linear(4, 3)
        self.x3 = nn.Linear(3, 2)
        self.x4 = nn.Linear(2, 1)
    def forward(self, x):
        x1 = nn.functional.relu(x)
        x2 = nn.functional.sigmoid(x1)
        x3 = nn.functional.tanh(x2)
        res1 = self.x1(x3)
        res2 = self.x2(x3)
        res3 = self.x3(x3)
        res4 = self.x4(x3)
        a = torch.cat([res1, res2], 3)
        b = torch.cat([res2, res3], 3)
        c = torch.cat([res3, torch.max(res4, res2)], 2)
        d = torch.cat([res4, res3], 3)
        v1 = [a, b, c, d]
        result = torch.cat(v1, 1)
        return result
# Inputs to the model
x = torch.randn(1, 5)
