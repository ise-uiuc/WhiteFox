
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(5, 10)
 
    def forward(t1):
        v1 = self.linear1(t1)
        v2 = v1 - other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5)
