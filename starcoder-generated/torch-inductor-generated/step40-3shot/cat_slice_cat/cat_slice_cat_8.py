
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = torch.cat(x1, dim=1)
        return v1[:, 0:9223372036854775807]
    
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 80, 80)
x2 = torch.randn(2, 79, 85)
