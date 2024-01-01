
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 16)
 
    def forward(self, x2):
        v1 = self.linear(x2)
        v2 = torch.sigmoid(v1)
        return v2

# Initializing the model
n = Model1()

# Inputs to the model
x2 = torch.randn(1, 3)
