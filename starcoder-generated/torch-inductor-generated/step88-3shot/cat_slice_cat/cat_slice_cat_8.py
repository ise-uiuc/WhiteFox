
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, t1, t2, t3):
        v1 = torch.cat([t1, t2, t3], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Generating dummy tensors to run through the model
t1 = torch.randn(1, 3)
t2 = torch.randn(1, 2)
t3 = torch.randn(1, 2)
