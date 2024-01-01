
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        v1 = torch.cat([x1, x2, x3], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:int(20 * 64 * 64 * 8 / 3)]
        v4 = torch.cat([v1, v3], dim=1)
        return v4
	
Inputs to the model
x1 = torch.randn(1, 768, 20, 64, 64)
x2 = torch.randn(1, 4096, 20, 64, 64)
x3 = torch.randn(1, 1, 20, 65, 65)
