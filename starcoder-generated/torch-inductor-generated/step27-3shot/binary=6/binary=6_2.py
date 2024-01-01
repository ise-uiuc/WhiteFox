
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.line = torch.nn.Linear(12, 21)

    def forward(self, x1, x2):
        v0 = torch.ones(1, 1)
        v1 = torch.matmul(v0, x1.transpose(0, 1)).transpose(0, 1)
        v2 = v0 * v1
        v3 = v2.flatten()
        v4 = torch.matmul(v3, x2) + 5
        v5 = torch.tanh(v4)
        v6 = v5.sum()
        return v6
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 12)
x2 = torch.randn(12)
