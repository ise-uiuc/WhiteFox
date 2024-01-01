
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
        self.other = torch.randn(8, )
 
    def forward(self, input):
        v1 = self.linear(input)
        v2 = v1 - self.other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
data = torch.randn(1, 3)
