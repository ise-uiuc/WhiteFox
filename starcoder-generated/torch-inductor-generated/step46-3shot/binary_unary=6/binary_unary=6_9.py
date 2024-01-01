
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2, bias=False)
        self.other = torch.nn.Parameter(data=torch.randn(2), requires_grad=True)
 
    def forward(self, x):
        x = self.linear(x)
        x = x - self.other
        x = F.relu(x)
        return x

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3)
