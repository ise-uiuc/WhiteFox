
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, input_tensors, size=None):
        v1 = torch.cat(input_tensors, dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:size]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Input to the model
input_tensors = [torch.randn(1, 256, 13, 13), torch.randn(1, 128, 26, 26), torch.randn(1, 64, 52, 52), torch.randn(1, 32, 104, 104), torch.randn(1, 128, 208, 208)]
size = 48
