
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(35, 15, bias=True)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = v1 - other
        return v2

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(1, 35, 1, 1)
