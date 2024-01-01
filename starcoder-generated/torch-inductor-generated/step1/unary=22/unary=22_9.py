
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 20)
 
    def forward(self, x):
        v1 = torch.tanh(self.fc(x))
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 10)
