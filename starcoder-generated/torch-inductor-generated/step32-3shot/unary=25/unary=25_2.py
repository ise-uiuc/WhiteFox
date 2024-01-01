
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 32, bias=True)
        self.lrelu = torch.nn.LeakyReLU(-0.5)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * -0.5
        v4 = torch.where(v2, v1, v3)
        return self.lrelu(v4)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(7, 8, 4, 4)
