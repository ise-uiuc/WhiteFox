
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x1):
        y1 = self.linear(x1)
 
        # Random variable
        z1 = torch.rand(1, 4)
 
        # Other value
        k1 = z1 + y1
        y2 = torch.cat((k1, y1), dim=1)
        y3 = torch.reshape(y2, (1, 16))
        y4 = torch.transpose(y3, 1, 2)
        y5 = torch.flatten(y4, start_dim=1)
        y6 = self.linear(y5)
        y7 = self.linear(y6)
 
        return y7

# Initializing the model
m = Model()
 
# Inputs to the model
x1 = torch.randn(2, 8)
