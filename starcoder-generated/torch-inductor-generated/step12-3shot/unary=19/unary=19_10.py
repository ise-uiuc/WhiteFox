
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Using 1 hidden layer (with 8 nodes) with sigmoid nonlinearity and ReLU nonlinearity
        self.linear1 = torch.nn.Linear(5, 8)
        self.linear2 = torch.nn.Linear(8, 1)
 
    def forward(self, x0):
        v0 = self.linear1(x0)
        v1 = torch.sigmoid(v0)
        v2 = v1 * 0.7
        v3 = v1 + v2
        v4 = torch.relu(v3)
        v5 = v4 + 1.5
        v6 = torch.softmax(v5, dim=-1)
        return v6, v1, v0

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn(1, 5)
__output__, ___, __ = m(x0)

