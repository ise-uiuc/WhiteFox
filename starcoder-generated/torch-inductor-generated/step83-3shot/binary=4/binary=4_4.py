
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(256, 10, bias=False)
 
        self.fc2 = torch.nn.Linear(256, 10, bias=False)
 
    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = self.fc2(x1)
        v3 = v1 + v2
        m = torch.nn.ReLU6()
        return m(v3)

# Initializing the model and the parameter list
m = Model()
__paramList__ = list(m.parameters())

# Generate the weight random tensor used in the model
__init_weight__ = torch.randn(10, 10)
__paramList__[0].data.copy_(__init_weight__)

# Inputs to the model
x1 = torch.randn(1, 256, requires_grad=True)
