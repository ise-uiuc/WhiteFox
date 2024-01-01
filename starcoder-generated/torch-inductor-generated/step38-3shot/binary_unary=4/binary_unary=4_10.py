
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        if torch.__version__ <= '1.8.0':
            self.linear = torch.nn.Linear(1024, 1000)
        else:
            self.linear = torch.nn.Linear(128, 384)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        v3 = torch.relu(v2)
        return v3

# Initializing the model
n = Model()

# Inputs to the model
x1 = torch.randn(1, 1024)
x2 = torch.randn(1, 128)
