
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 5)
        self.linear2 = torch.nn.Linear(5, 10)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = self.linear2(v1)
        return v2 * 0.75

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
