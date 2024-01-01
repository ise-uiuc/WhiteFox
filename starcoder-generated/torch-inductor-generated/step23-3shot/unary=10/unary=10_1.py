
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        v1 = x1.flatten(start_dim=1)
        v2 = torch.tensor([2.67550209e-05, 8.79326090e-05, 4.33683297e-05, 2.56617851e-05,
                    1.62987153e-05, 1.14969637e-05, 8.48186018e-06, 6.51233520e-06], requires_grad=True)
        v3 = torch.matmul(v1, v2) + 3
        v4 = torch.clamp_min(v3, 0)
        v5 = torch.clamp_max(v4, 6)
        v6 = v5 / 6
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16, 1, 1)
