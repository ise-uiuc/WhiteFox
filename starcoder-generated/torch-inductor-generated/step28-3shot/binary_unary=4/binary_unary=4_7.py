
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 20)
 
    def forward(self, x2, other):
        v1 = self.linear(x2)
        v2 = v1 + other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Create another tensor to add
other = torch.randn(20, 500)

# Inputs to the model
x2 = torch.randn(10, 500)
