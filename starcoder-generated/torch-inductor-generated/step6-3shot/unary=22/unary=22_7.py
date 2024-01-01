  
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(224, 1024)
 
    def forward(self, x1):
        o0 = self.linear(x1)
        o1 = torch.tanh(o0)
        return o1

# Initializing model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 224)
