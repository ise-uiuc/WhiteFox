
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(30, 20)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + torch.ones(v1.size())
        return v2
    
# Initializing the model
m1 = Model1()

# Inputs to the model
x = torch.randn(1, 30)
