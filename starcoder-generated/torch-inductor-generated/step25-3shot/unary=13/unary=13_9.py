
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(4, 4)
        self.sigmoid = torch.nn.Sigmoid()
 
    def forward(self, x1, x2):
        v1 = self.linear1(x1)
        v2 = self.sigmoid(v1)
        v2 = v1 * v2
        return v2

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(4)
x2 = torch.randn(4)
