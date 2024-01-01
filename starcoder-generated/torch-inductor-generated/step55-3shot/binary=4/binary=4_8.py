
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 256)
        
    def forward(self, x, other):
        v1 = self.linear(x)
        v2 = v1 + other
        return v2

# Initializing model 
m = Model()

# Input to the model
x = torch.randn(2, 5, 2, 2)
other = torch.randn(2, 256, 2, 2)
