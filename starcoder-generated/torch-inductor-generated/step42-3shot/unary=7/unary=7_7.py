
class Model(torch.nn.Linear):
    def __init__(self):
        super().__init__(64, 64)
 
    def forward(self, x1):
        l1 = super().forward(x1)
        l2 = l1 * torch.clamp(l1.min(), 0, 6) + 3
        l3 = l2 / 6
        return l3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
