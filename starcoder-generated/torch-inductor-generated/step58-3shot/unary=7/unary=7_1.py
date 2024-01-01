
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 8, bias=False)
 
    def forward(self, x1):
        x2_1 = self.linear(x1)
        x2_2 = torch.clamp(x2_1, min=0, max=6)
        x2_3 = x2_2 + 3
        x2_4 = x2_3 / 6
        v1 = x2_1 + x2_4
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 2)
