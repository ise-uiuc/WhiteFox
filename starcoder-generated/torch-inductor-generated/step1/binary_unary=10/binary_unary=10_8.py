
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(20, 10)
 
    def forward(self, x, param=None):
        v1 = self.linear(x).tanh()
        if param is not None:
            return (v1 + param).relu()
        else:
            return v1.relu()

# Initializing the model:
m = Model()

# Inputs to the model
x = torch.randn(1, 20)
