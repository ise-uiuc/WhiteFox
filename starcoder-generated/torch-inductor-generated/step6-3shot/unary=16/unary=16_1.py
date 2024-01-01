
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(40, 80)
 
    def forward(self, x1):
        y1 = self.linear(x1)
        y2 = F.relu(y1)
        return y2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 40)
