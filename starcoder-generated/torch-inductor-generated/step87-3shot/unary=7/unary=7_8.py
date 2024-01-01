
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, t7):
        t8 = self.linear(t7)
        t9 = t8 * F.hardtanh(t8, 0, 6) + 3
        t10 = t9 / 6
        return t10

# Initializing the model
m = Model()

# Inputs to the model
t7 = torch.randn(1, 3)
