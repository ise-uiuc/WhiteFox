
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 8)
 
    def forward(self, x2, x3):
        v1 = self.linear(x2)
        return v1 - x3

m = Model()

# Inputs to the model
x2 = torch.randn(1, 10)
x3 = torch.randn(20)
