
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
 
    def forward(self, x):
        t1 = x
        t2 = self.linear(t1, b=other)
        t3 = F.relu(t2)
        return t3

# Initializing the model
m = Model(other=torch.randn(1, 16))

# Inputs to the model
x = torch.randn(1, 16)
