
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 8, bias=True)
 
    def forward(self, x1):
        q1 = self.linear(x1)
        q2 = q1 * (q1.clamp(0, 6) + 3)
        return q2 / 6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
