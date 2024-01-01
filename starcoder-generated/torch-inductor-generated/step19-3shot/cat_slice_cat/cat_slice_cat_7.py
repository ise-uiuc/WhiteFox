
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_list):
        x1 = input_list[0]
        x2 = x1 + 1
        x3 = x1 * 0.7071067811865476
        x4 = torch.erf(x3)
        x5 = x4 * 0.8412536529812128
        x6 = x2 * x5
        x7 = x1 + 1
        x8 = 0.7071067811865476 * x7
        x9 = torch.erf(x8)
        x10 = 0.8412536529812128 * x9
        y0 = 1.0
        return [ x6, x10 ]

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1000)
