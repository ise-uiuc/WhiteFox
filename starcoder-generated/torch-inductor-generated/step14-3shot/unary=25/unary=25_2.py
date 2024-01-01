
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.act = torch.nn.LeakyReLU(negative_slope=0.0225)
 
    def forward(self, x1):
        v1 = self.act(x1)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
