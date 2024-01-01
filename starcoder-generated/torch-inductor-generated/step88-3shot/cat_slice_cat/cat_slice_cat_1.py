
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        v1 = torch.cat([x1, x2], dim=1)
        v1_shape = v1.shape
        v2 = v1[:, 0:v1_shape[1]]  # Slice the tensor along dimension 1
        v4 = torch.cat([x1, v2], dim=1)  # Concatenate the original concatenated tensor and the sliced tensor along dimension 1
        return v1, v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 1, 128, 128)
x2 = torch.randn(2, 2, 128, 128)
