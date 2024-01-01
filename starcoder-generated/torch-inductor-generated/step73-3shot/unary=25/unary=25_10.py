
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 100, bias=False)
        self.negative_slope = 0.01
 
    def forward(self, input):
        y = self.linear(input)
        return torch.where(y > 0, y, y * self.negative_slope)
 
# Initializing the model
m = Model()
 

# Inputs to the model
x = torch.randn(1, 100)
