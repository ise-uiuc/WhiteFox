
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 20)
        self.linear2 = torch.nn.Linear(20, 4)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)