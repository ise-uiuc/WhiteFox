
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(10, 20)

    def forward(self, x1, **kwargs):
        v1 = self.lin(x1)
        return v1 + kwargs.get('other', None)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 10)
