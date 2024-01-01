
class Model(nn.Module):
    def __init__(self, in_size, out_size):
        super(Model, self).__init__()
        self.l1 = nn.Linear(in_size, out_size)
 
    def forward(self, input, other):
        v1 = self.l1(input)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model(10, 100)

# Inputs to the model
input = torch.randn(10)
other = torch.arange(100).float()
