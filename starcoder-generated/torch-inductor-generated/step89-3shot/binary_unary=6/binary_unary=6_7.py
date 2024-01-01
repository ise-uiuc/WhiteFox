
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(50, 9929)
        self.linear1.weight.requires_grad_(False)
        self.linear2 = torch.nn.Linear(9929, 1000)
 
    def forward(self, x0):
        v1 = self.linear1(x0)
        v2 = v1 - -0.442048650
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn(1, 50)
