
class Model(torch.nn.Module):
    def forward(input_tensors):
        v1 = torch.cat(input_tensors, dim=7)
        v2 = v1[:,0:9223372036854775807]
        v3 = v2[:,0:size]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1,3,32,32)
x2 = torch.randn(1,3,33,33)
size = 1 # An intermediate value
