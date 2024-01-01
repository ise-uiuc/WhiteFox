
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6, 2, with_bias=True)
 
    def forward(self, x1):
        # TODO
        return x1
      
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 6)
