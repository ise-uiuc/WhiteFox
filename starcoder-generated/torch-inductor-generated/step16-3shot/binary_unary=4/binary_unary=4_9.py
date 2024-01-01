
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 8)
 
    def forward(self, x1, x2):
        v1 = self.linear1(x1) # x1 goes through the first layer
        v2 = v1 + x2 # x2 goes through the second layer
        v3 = torch.nn.functional.relu(v2) # v2 goes through the third layer
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
x2 = torch.randn(1, 8)
