
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(256, 256)
 
    def forward(self, x1, x2):
        if (x1 < 0):
            v1 = self.linear(x2)
            v2 = v1 + x1
        elif (x1 > 0):
            v3 = self.linear(x2)
            v4 = v3 + x1
        else:
            v5 = self.linear(x2)
            v6 = v5 + x1
        return v6
    
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 256, 1, 1)
x2 = torch.randn(1, 256, 1, 1)
