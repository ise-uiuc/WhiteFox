
class ResBlock(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_size, out_size)
        self.linear2 = torch.nn.Linear(out_size, out_size)
 
    def forward(self, x1, other):
        v1 = self.linear1(x1)
        v2 = v1 + other
        v3 = self.linear2(v2)
        return v3

# Initializing the model
m = ResBlock(10, 20)

# Inputs to the model
x1 = torch.randn(10)
__other__ = torch.randn(20)
