
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 3)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - torch.tensor([0.31499201, 0.78615003, 1.2662738 ], requires_grad=True)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
