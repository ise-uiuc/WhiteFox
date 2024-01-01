
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linearrelu = torch.nn.Sequential(
            torch.nn.Linear(5, 10),
            torch.nn.ReLU()
        )
 
    def forward(self, x1, other=None):
        x2 = self.linearrelu(x1)
        if other is not None:
            x2 = x2 + other
        return x2

# Initializing the model
m = Model()

# Inputs to the model
