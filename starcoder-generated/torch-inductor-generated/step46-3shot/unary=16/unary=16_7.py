
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = torch.nn.Linear(3, 32)
        self.net2 = torch.nn.ReLU()
 
    def forward(self, x1):
        v1 = self.net1(x1)
        v2 = self.net2(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 5, 5)
