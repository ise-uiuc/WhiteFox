
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10)
 
    def forward(self, x1):
        y1 = self.linear(x1)
        return y1 - 3
        
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5)
