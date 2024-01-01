
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6, 2)
 
    def forward(self, x1, output_activation=True):
        v1 = self.linear(x1)
        v2 = v1 + x1
        if output_activation:
            v3 = v2.relu()
        else:
            v3 = v2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 6)
