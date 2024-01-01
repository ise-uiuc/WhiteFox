
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(5, 1)
        self.sigmoid = torch.nn.Sigmoid()
 
    def forward(self, x2):
        v1 = self.linear_1(x2)
        v2 = self.sigmoid(v1)
        v3 = v1 * v2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 5)
