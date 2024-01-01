
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.ReLU6(),
            torch.nn.Linear(3, 4),
            torch.nn.Linear(4, 5)
        )
    
    def forward(self, x):
        return self.model(x)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(5, 3)
