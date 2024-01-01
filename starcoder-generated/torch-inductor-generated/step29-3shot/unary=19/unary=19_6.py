
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
 
    def forward(self, p1):
        v1 = self.linear(p1)
        v2 = torch.sigmoid(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
p1 = torch.randn(250, 3)
