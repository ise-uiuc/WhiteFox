
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(8, 1)
 
    def forward(self, x2):
        x = torch.sigmoid(self.linear(x2))
        return x

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 8)
