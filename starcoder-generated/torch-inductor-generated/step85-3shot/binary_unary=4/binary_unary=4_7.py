
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module = torch.nn.TransformerEncoder()
 
    def forward(self, x1, x2, x3):
        v1 = self.module(x2, x3)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 10)
x2 = torch.randn(1, 10, 2)
x3 = torch.tensor([[1]])
