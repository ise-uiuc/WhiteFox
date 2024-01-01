

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_ = torch.nn.Linear(64, 1)
 
    def forward(self, x2):
        v2 = self.linear_(x2)
        v3 = torch.sigmoid(v2)
        return v3

# Initializing the model
m = Model()

# Input to the model
x2 = torch.randn(1, 64)
