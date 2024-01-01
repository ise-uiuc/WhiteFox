
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(256 * 256 + 3, 768)

    def forward(self, input_tensor, other):
        t1 = self.linear(input_tensor)
        t2 = t1 + other
        return t2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(700, 256 * 256 + 3)
x2 = torch.randn(700, 768)
