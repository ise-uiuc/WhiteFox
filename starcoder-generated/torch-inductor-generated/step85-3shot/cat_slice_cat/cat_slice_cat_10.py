
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = torch.cat([input_tensor1, input_tensor2], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:128]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)
x2 = torch.randn(1, 13, 64, 64)
