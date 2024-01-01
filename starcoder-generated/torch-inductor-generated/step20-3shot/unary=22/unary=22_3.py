
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8, 1)
 
    def forward(self, x1):
        y1 = self.fc(x1)
        y2 = torch.tanh(y1)
        return y2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8, 4, 8, 8)
