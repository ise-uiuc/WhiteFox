
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 14)
 
    def forward(self, x, y):
        z = self.linear(x)
        z = z + y
        return z
    
# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8)
y = torch.randn(1, 14)
