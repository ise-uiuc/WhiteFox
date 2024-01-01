
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = torch.nn.Linear(3, 6)
        self.dense2 = torch.nn.Linear(6, 19)
 
    def forward(self, x):
        v1 = self.dense1(x)
        v2 = self.dense2(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3)
