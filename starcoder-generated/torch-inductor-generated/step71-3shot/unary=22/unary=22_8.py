
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l = torch.nn.Linear(in_features=2, out_features=3, bias=False)

    def forward(self, x):
        x = self.l(x)
        x = torch.tanh(x)
        return x

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randint(low=-10, high=10, size=(3, 2))
