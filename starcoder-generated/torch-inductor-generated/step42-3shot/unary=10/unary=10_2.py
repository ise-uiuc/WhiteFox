
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
         
        self.linear = torch.nn.Linear(8, 16)
 
    def forward(self, x1):
 
          # Applies l2 = 3
          l1 = self.linear(x1)
          l2 = l1 + 3
 
          # Applies l5 = l4 / 6
          l3 = torch.clamp_min(l2, 0) # Clamp the output of the addition operation to a minimum of 0
          l4 = torch.clamp_max(l3, 6) # Clamp the output of the previous operation to a maximum of 6
          l5 = l4 / 6 # Divide the output of the previous operation by 6
          return l5 # The output is in l5


# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
