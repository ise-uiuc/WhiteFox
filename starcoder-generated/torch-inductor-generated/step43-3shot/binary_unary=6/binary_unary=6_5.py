
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(1024, 128, bias=False),
            torch.nn.ReLU()
        )
 
    def forward(self, x1):
        v1 = self.layers(x1)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1024)
