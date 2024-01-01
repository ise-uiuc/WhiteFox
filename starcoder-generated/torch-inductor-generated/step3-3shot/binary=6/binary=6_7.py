
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        output = v1 - 5.0
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
