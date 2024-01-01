
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x1):
        x = torch.cat(x1, x1, 1)
        x = x.view(2, 4)
        x += 1
        return x.relu()

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 2)
