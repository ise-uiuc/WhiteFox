
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
 
    def forward(self, x1):
        y2 = self.linear(x1)
        y3 = torch.clamp(y2, min=0, max= 6)
        y4 = y3 +3
        return y4 / 6

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(1, 16) 

