
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 5, bias=True)
 
    def forward(self, x):
        out = torch.nn.functional.leaky_relu(input=x, negative_slope=0.1, inplace=False)
        return out

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 4)
