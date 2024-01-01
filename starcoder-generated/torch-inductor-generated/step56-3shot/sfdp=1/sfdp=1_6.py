
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.W = torch.nn.Linear(21, 100)
 
    def forward(self, x1):
        v1 = self.W(x1)
        v2 = v1.matmul(x1.transpose(0, 1))
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(5, 21)
