
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
 
    def forward(self, v1, v2):
        t1 = self.linear(v1)
        t2 = t1 - v2
        return t2

# Initializing the input tensors and the model
v1 = torch.randn(1, 2)
v2 = torch.randn(1, 3)
m = Model()

# Inputs to the model
