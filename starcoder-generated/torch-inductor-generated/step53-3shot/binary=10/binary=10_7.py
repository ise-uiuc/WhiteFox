
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(18, 16, bias=True)
 
    def forward(self, input, other):
        v1 = self.linear(input)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(2, 18)
other = torch.randn(2,16)
