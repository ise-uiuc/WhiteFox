
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.clone()
        x = x.repeat(2, 1, 1)

        y = x.reshape(x).shape[2]
        z = x.view(2, x.shape[0] * x.shape[1], x.shape[2] * x.shape[3])
        # Note the `y` must be computed before `z`
        # This is to make sure that it is valid graph
        return z
# Inputs to the model
x = torch.randn(2, 3, 4)
