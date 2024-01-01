
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 16)
        self.sigmoid = torch.nn.Sigmoid()
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.relu(v1)
        v3 = self.sigmoid(v1)
        v4 = torch.tanh(v1)
        v5 = torch.softmax(v1)
        v6 = torch.sigmoid(v1)
        return (v2, v3, v4, v5, v6)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
