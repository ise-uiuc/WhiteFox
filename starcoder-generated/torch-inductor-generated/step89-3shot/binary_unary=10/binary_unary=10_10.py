
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 512)
        self.linear2 = torch.nn.Linear(512, 512)
 
    def forward(self, x, y):
        v1 = self.linear1(x)
        v2 = self.linear2(y)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3)
y = torch.randn(1, 3)
