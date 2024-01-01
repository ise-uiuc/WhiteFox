
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 32, bias=True)

    def forward(self, x1):
        # TODO: Add forward function code
        pass

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
