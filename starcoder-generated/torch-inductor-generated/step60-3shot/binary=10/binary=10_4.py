
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(25 * 25, 128)
 
    def forward(self, x1):
        v1 = x1.view((x1.size()[0], -1))
        v2 = self.linear(v1)
        return v2 + x1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 25, 25)
