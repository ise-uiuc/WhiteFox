
def __get_model(other):
    m = Model()
    class __Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x1):
            return m(x1) + other
    return __Wrapper

# Initializing the model
other = torch.randn(8, 3, 64, 64)
m = __get_model(other)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
