
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8):
        v1 = torch.cat([x1, x2])
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:476531860]
        v4 = torch.cat([v1, v3])
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 47053644, 76577761)
x2 = torch.randn(1, 98048153, 8736343)
x3 = torch.randn(1, 10, 76596)
x4 = torch.randn(1, 84, 34987)
x5 = torch.randn(1, 62, 904767)
x6 = torch.randn(1, 23090445, 4756657)
x7 = torch.randn(1, 28600, 48567)
x8 = torch.randn(1, 134987987657, 73564297654)
