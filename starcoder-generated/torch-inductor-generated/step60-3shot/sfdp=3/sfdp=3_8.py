
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Transformer()
 
    def forward(self, x1, x2, x3):
        v1 = self.model(x1, x2, x3)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 6, 512)
x2 = torch.randn(1, 6, 512)
x3 = torch.randn(1, 6, 512)
