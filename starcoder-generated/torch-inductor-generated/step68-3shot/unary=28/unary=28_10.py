
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 15)
 
    def forward(self, x):
        v1 = self.linear(x)
        self.linear.clamp_min_(min=0)
        v2 = self.linear(x)
        self.linear.clamp_max_(max=5)
        v3 = self.linear(x)
    
# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(50, 10)
