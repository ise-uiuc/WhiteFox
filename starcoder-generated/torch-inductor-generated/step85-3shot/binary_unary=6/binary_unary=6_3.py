
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 32)
        self.act = torch.nn.ReLU()
 
    def forward(self, x1):
        x2 = self.linear1(x1)
        x3 = x2 - 0.3
        x4 = self.act(x3)
        return x4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)
