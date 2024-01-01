
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Conv2d(5, 3, 1)
        self.linear2 = torch.nn.Linear(3, 5)
 
    def forward(self, x):
        v1 = self.linear1(x)
        v2 = torch.relu(v1)
        v3 = self.linear2(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 5, 5, 5)
