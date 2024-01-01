
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    # BMM
    def forward(self, x1, x2):
        if conditional_func(x1, x2):
            # First argument is used as a bias tensor after bmm() operation.
            v3 = torch.bmm(x1.permute(0, 2, 1), x2)
        else:
            # Second argument is used as a bias tensor after bmm() operation.
            v4 = torch.bmm(x1, x2.permute(0, 2, 1))
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
