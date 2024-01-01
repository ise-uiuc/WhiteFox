
class Model(nn.Module):
    def __init__(self, negative_slope=0.2):
        super(Model, self).__init__()
        self.linear = nn.Linear(3, 4)
        self.negative_slope = negative_slope
 
    def forward(self, input):
        x = self.linear(input)
        t2 = x > 0
        t3 = x * self.negative_slope
        x = torch.where(t2, x, t3)
        return x

# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(64, 3)
