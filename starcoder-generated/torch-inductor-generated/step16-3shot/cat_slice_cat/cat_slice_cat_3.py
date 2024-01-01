
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x14, x24):
        v14 = torch.cat([x14, x24], dim=1)
        v24 = v14[:, 0:9223372036854775807]
        v34 = v24[:, 0:57344]
        v44 = torch.cat([v14, v34], dim=1)
        return v44

# Initializing the model
m = Model()

# Initializing input tensors
x14 = torch.randn(1, 64, 3, 8)
x24 = torch.randn(1, 64, 3, 8)

# Inputs to the model
