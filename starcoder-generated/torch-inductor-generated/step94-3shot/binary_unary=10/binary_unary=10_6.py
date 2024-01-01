
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(5,10)
        self.linear2 = torch.nn.Linear(10,15)
        
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = self.linear2(v1)
        v3 = v2 + x1
        v4 = torch.relu(v3)
    
        return v4
    
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5)
x2 = torch.randn(1, 15)
