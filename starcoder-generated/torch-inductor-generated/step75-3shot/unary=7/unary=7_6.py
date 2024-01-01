
class Model(torch.nn.Module):
    def __init__(self):
    	super().__init__()
    	self.linear = torch.nn.Linear(16, 1)
 
    def forward(self, input):
      v1 = self.linear(input)
      v2 = torch.clamp(v1, min=0, max=6) + 3
      v3 = v2 / 6
      return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 9, 16)
