
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(20, 10)
        self.linear2 = torch.nn.Linear(10, 30)
 
    def forward(self, x):
        v1 = self.linear1(x)
        v2 = v1 + x
        v4 = self.linear2(relu(v2))
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 20)
