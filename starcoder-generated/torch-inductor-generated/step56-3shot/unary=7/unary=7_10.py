
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 16)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0, inplace=False)
        self.linear2 = torch.nn.Linear(16, 1)
 
    def forward(self, x1):
        l1 = self.linear1(x1)
        l2 = l1 * self.lrelu(l1 + 3).clamp(min=0, max=6)
        out = l3 / 6
        return out

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
