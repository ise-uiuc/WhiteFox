
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.BatchNorm1d(100),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU()
        )
        
    def forward(self, x):
        return self.linear(x)
        
# Initializing the model
__m__ = Model()

# Inputs to the model
x = torch.randn(100)
