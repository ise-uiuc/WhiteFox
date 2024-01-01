
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10)

    def forward(self, x):
        h = self.linear(x)
        a = torch.sigmoid(h)
        return a

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 5)
# Outputs from the model
y = m(x)

