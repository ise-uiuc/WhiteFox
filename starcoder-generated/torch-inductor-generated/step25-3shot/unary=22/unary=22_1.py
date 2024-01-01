
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6, 10, bias=True)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.tanh(v1)
        return v2

# Initializing the model
m = Model()
# Setting random seeds
torch.manual_seed(0)

# Inputs to the model
x1 = torch.randn(1, 6)
