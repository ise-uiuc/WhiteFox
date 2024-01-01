
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2, x3):
        v1 = torch.cat((x1, x2, x3), dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, :7]
        v4 = torch.cat((v1, v3), dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
__input_tensor_1__ = torch.randn(1, 100, 1, 1)
__input_tensor_2__ = torch.randn(1, 8, 1, 1)
__input_tensor_3__ = torch.randn(1, 15, 1, 1)
