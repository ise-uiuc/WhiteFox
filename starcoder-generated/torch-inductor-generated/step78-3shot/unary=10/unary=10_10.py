
class Model(torch.nn.Module):
    def __init__ (self):
        super().__init__()
        self.linear = torch.nn.Linear(5,10)
 
    def forward(self, input):
        X = self.linear(input)
        X = X + 3
        X = torch.clamp_max(torch.clamp_min(x,0),6)
        X = X / 6
        return X

# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(128,5)
o1 = m(input)

