
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 16)
 
    def forward(self, x1):
        x = self.linear(x1)
        x = x - other
        x = relu(x)
        return x
    
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32)
