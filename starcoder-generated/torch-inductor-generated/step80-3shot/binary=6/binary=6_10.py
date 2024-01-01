
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 50)
        self.linear2 = torch.nn.Linear(50, 10)
 
    def forward(self, x1):
        x2 = self.linear1(x1)
        x3 = self.linear2(x2)
        return (x2, x3)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(64, 10)
x2 = torch.randn(64, 50)
