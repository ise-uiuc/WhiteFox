
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight1 = Parameter(torch.randint(1, 10, (8, 8)), torch.float32)
        self.weight2 = Parameter(torch.randint(1, 10, (8, 8)), torch.float32)
        self.weight3 = Parameter(torch.randint(1, 10, (8, 8)), torch.float32)
    def forward(self, x):
        if x.sum() <= 2:
            out = torch.cat([x, self.weight1, self.weight1, self.weight1, self.weight1, self.weight1, self.weight1, self.weight2], 1)
        elif x.sum() <= 8:
            out = torch.cat([self.weight3, self.weight1, self.weight2], 1)
        else:
            out = torch.cat([self.weight1, self.weight2], 1)
        return out
# Inputs to the model
x = torch.randn(1, 8)
