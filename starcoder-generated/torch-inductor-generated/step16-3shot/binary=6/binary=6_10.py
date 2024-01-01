
class Model(torch.nn.Module):
    def __init__(self, in_shape: int=512, out_shape: int=1152, num: int=512, bias: bool=False):
        super().__init__()
        self.dense1 = torch.nn.Linear(in_shape, out_shape, bias=bias)
        self.dense2 = torch.nn.Linear(out_shape, num, bias=bias)

    def forward(self, x):
        v1 = torch.flatten(x, start_dim=1)
        v2 = self.dense1(v1)
        v3 = v2 - 0.5
        v4 = self.dense2(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 258, 258)
