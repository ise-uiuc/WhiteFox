
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #self.linear = torch.nn.Linear(16, 16)
        self.linear = MyLinear()
        
    def forward(self, x, other=None):
        if other is None:
            other = torch.randn_like(x)
        v1 = self.linear(x)
        v2 = v1 + other
        return v2    
    
# Initializing the model
m = Model()
# Inputs to the model
x = torch.randn(1, 16)
