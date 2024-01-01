
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_1 = torch.nn.Linear(3 * 3 * 3, 64)
        self.linear_2 = torch.nn.Linear(64, 2)
        
 
    def forward(self, x1):
        f1 = self.flatten(x1)
        v1 = self.linear_1(f1)
        v2 = torch.tanh(v1)
        v3 = self.linear_2(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 3, 3)
