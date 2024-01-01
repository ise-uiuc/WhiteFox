
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 128)
        self.relu = torch.nn.ReLU()
 
    def forward(self, x):
        l1 = self.linear(x)
        l2 = l1 + 3
        l3 = self.relu(l2)
        l4 = self.relu(l3)
        l5 = l4 / 6
        return l5

# Initializing the model
m = Model()

# Inputs to the model
x = torch.rand(1, 128)
