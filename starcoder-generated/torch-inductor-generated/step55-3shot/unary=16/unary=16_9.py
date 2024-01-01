
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 10)
 
    def forward(self, x2):
        v1 = self.linear(x2)
        v2 = F.relu(v1, inplace=True)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(64, 128)
