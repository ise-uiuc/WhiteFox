
class Model():
    def __init__(self, min_value=0, max_value=1):
         self.lin = torch.nn.Linear(3, 7)
  
         self.min_value = min_value
         self.max_value = max_value
  
    def forward(self, x1):
         t1 = self.lin(x1)
         t2 = torch.clamp_min(t1, self.min_value)
         t3 = torch.clamp_max(t2, self.max_value)
         return t3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
