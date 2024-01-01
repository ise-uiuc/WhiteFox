
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(4, 8)
 
    def forward(self, x2):
        v7 = self.l1(x2)
        v8 = v7.relu()
        return v8

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 4)
