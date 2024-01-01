
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(other.shape[1], 128)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other
        return v2

# Initialize the model
other = torch.randn(1, 128)
m = Model(other)

# Initialize the inputs to the model
x1 = torch.randn(1, other.shape[1])
