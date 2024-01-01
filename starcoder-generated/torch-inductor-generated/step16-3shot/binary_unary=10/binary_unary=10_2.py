
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(128, 32)
        self.linear2 = torch.nn.Linear(32, 16)
        
    def forward(self, x0):
        v1 = self.linear1(x0)
        v2 = self.linear2(v1)  
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn(1, 128)
