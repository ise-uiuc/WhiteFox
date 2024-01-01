
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
 
    def forward(self, i):
        l1 = self.linear(i)
        l2 = l1 * torch.clamp(
            input = (l1 + 3),
            min = 0,
            max = 6
        )
        l3 = l2 / 6
        return l3

# Initializing the model
m = Model()

# Inputs to the model
i = torch.randn(1, 3)
