
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = torch.cat([x1, x1, x1, x1, x1], 1) 
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:__SIZE__]
        v5 = torch.cat([v1, v3], 1)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, __NUM_CHANNELS__, __SIZE__, __SIZE__)
