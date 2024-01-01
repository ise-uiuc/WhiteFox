
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        l1= linear(input)
        l2 = l1 * clamp(min=0, max=6, l1 + 3)
        l3 = l2 / 6
        return l3

# Initializing the model
m = Model()

# Input to the model
x = torch.randn(1, 10)
