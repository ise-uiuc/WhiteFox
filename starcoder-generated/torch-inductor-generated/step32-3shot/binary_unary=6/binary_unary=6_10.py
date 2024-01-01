
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(30, 784)
        self.other = other
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 - self.other
        v3 = torch.nn.ReLU()(v2)
        return v3

# Initializing the model
other = 0.5
num_hidden = 10
batch = 16
m = Model(other)

# Inputs to the model
x1 = torch.randn(batch, num_hidden)
x2 = torch.randn(batch, num_hidden)
