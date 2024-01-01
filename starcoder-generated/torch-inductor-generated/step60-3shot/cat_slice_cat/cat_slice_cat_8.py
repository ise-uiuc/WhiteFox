
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
  
    def forward(self, x1):
	
        # t1 = torch.cat([x1, x2], dim=1)
        v1 = x1[:, 0:9223372036854775807]
        # t2 = t1[:, 0:8388608]
        v2 = v1[:, 0:size]
        # t3 = torch.cat([t1, t2], dim=1)
        v3 = torch.cat([x1, v2], 1)
      
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128, 124, 124)
# x2 = torch.randn(1, 128, 12, 12)
