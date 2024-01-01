
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.negative_slope = negative_slope
        self.linear = torch.nn.Linear(1, 1, bias=False)
 
    def forward(self, input):
        z = self.linear(input)
        return torch.where((z > 0), z, self.negative_slope * z).view(1, 1)

# Initializing the model
s = 0.01
m1 = Model(s)
m2 = torch.nn.LeakyReLU(s, inplace=False)

# Inputs to the model
x = torch.randn(1, 1)
