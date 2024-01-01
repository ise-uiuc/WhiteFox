
class Model(torch.nn.Module):
   def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
   def forward(self, input, other):
        t1 = self.linear(input).transpose(0, 1)
        t2 = t1 + other.transpose(0, 1)
        t3 = torch.relu(t2)
        return t3.transpose(0, 1)

# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(8, 8)
other = torch.randn(8, 8)
