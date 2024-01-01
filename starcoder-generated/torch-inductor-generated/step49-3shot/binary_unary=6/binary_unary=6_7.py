
class Model(torch.nn.Module):
    __constants__ = ['other']
 
    def __init__(self, output_size, other):
        super().__init__()
        self.other = other
        self.linear = torch.nn.Linear(12, output_size)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - self.other
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model(6, 0.7463192030470237)

# Inputs to the model
x1 = torch.randn(1, 12)
