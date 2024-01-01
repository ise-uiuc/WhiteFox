
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6, 4)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = F.sigmoid(v1)
        v3 = v1 * v2
        return v3

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(1, 6)
