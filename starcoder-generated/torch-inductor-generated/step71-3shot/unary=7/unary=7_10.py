
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(3, 8)
        self.linear_2 = torch.nn.Linear(8, 6)
 
    def forward(self, x1):
        l1 = self.linear_1(x1)
        l2 = self.linear_2(f.relu(l1))
        l3 = l2 / 6
        return l3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
