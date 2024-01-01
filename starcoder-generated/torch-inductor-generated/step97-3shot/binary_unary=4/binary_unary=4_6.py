
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 32)
 
    def forward(self, x1, other=torch.ones(32)):
        y1 = self.linear(x1)
        y2 = y1 + other
        y3 = torch.relu(y2)
        return y3
    
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32)
