
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, input_tensors1):
        v1 = torch.cat(input_tensors1, dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:size]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
input_tensors1 = []
for _ in range(2):
    input_tensors1.append(torch.randn(1, 3, 64, 64))
