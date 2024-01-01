
class Model(torch.nn.Module):
    def __init__(self):
       super().__init__()
       self.linear = torch.nn.Linear(128, 256)
 
    def forward(self, input, other):
        x1 = self.linear(input)
        x2 = x1 + other
        x3 = F.relu(x2)
        return x3

# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(1, 128)
other = torch.randn(1, 256)
