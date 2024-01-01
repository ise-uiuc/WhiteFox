
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 5)
        self.bn = torch.nn.BatchNorm1d(15)
    def forward(self, x1, x2):
        x1 = self.embedding(x1)
        x2 = self.bn(x2)
        return torch.cat([x1, x2], -1)
# Inputs to the model
x1 = torch.randint(9, (4,))
x2 = torch.randn(4, 3, 5)
