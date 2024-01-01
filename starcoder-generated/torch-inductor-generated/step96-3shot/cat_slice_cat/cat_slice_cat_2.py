
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        v1 = torch.cat(input_tensors, dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:size]
        return torch.cat([v1, v3], dim=1)
# Initializing the model
m = Model()
# Inputs to the model
x1_shape = (1, 3, 64, 64)
x2_shape = (1, 3, 8, 64)
x3_shape = (1, 4, 8, 64)
x4_shape = (1, 1, 32, 32)
x1 = torch.randn(x1_shape)
x2 = torch.randn(x2_shape)
x3 = torch.randn(x3_shape)
x4 = torch.randn(x4_shape)
