
class Model(torch.nn.Module): 
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
    def forward(self, x1):
        return self.linear(x1) - 2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 10)
