
class Model():
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
        self.other = other
 
    def forward(self, x3):
        v1 = self.linear(x3)
        v2 = v1 + self.other
        v3 = torch.relu(v2)
        return v3

# Initializing the model with an additional input 'other'
other = torch.ones(8)
m = Model(other=other)

# Inputs to the model
x3 = torch.randn(3, 3)
