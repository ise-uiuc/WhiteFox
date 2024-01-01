
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
    def _init_weights(self):
        torch.nn.init.kaiming_normal_(self.linear.weight, nonlinearity='sigmoid')
    def forward(self, x1):
        return self.linear(x1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
