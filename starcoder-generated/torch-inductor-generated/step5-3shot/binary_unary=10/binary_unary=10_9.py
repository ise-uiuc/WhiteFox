
class Model(torch.nn.Module):
    def __init__(self, size1):
        super().__init__()
        self.linear = torch.nn.Linear(size1, size1 * 2)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + x1
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model(10)

# Inputs to the model
x1 = torch.randn(1, 10)
