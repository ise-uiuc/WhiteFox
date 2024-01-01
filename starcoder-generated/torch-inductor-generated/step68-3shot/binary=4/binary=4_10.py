
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(50, 10)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other
        return v2

# Initializing the model
torch.manual_seed(np.random.randint(1000))
m = Model()
m.linear.weight.data = torch.randn((10, 50))
m.linear.bias.data = torch.randn(10)

# Inputs to the model
x1 = torch.randn(1, 50)
other = torch.randn(1, 10)
