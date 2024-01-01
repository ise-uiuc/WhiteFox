
class Model(torch.nn.Module):
    def __init__():
        super().__init__()
        self.fc = torch.nn.Linear(100, 100)
 
    def forward(self, x):
        v1 = self.fc(x)
        return v1 - 5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 100)
