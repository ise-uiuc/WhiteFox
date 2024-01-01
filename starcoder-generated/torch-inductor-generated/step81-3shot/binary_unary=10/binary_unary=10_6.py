
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(3, 5)
        self.linear_2 = torch.nn.Linear(3, 5)
 
    def forward(self, x1, x2):
        v1 = self.linear_1(x1)
        v2 = self.linear_2(x2)
        v3 = v1 + v2
        v4 = F.relu(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
x2 = torch.randn(1, 3)
