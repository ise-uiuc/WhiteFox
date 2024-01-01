
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x):
        v1 = torch.cat(input_tensors=x, dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:v1.shape[1] - x.shape[1] + 1]
        return torch.cat([v1, v3], dim=1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 8, 32)
