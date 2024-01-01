
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(128, 64, bias=False)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - other
        v3 = torch.relu(v2)
        return v3

# Other value used in the model.
other = 1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)
