
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 4)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        return v1 + other

# Initializing the model and the input tensor
m = Model()
x1 = torch.randn(1, 8)
