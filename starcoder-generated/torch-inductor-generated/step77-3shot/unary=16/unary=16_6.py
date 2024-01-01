
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
 
    def forward(self, x3):
        y1 = self.linear(x3)
        y1 = F.relu(y1)
        return y1

# Initializing the model
m = Model()

# Inputs to the model
x3 = torch.randn(1, 2)
