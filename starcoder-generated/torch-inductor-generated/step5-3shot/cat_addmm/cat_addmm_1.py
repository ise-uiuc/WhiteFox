
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 8)

    def forward(self, x1):
        l1 = self.linear(x1)
        l2 = torch.rand(l1.shape)
        l3 = torch.cat([l1,l2], 1)
        return l3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8, 16)
