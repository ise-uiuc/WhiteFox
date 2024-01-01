
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(20, 20)
 
    def forward(self, x1):
        v1 = self.layer1(x1)
        v2 = v1 - torch.ones(v1.size())
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 20)
