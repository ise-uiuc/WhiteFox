
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 100)
        self.linear1 = torch.nn.Linear(100, 100)
        self.linear2 = torch.nn.Linear(100, 100)

    def forward(self, x, relu):
        m = self.linear(x)
        if relu:
            m = F.relu(self.linear1(m))
        else:
            m = self.linear2(m)
        m = torch.clamp(m, min=0, max=6)
        m = m / 6
        return m

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 100)
relu = False
