
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        u1 = [x1, x2, x3]
        u2 = torch.cat(u1, dim=1)
        u3 = u2[:, 0:9223372036854775807]
        u4 = u3[:, 0:16777216-u3.shape[1]]
        u5 = [u2, u4]
        return torch.cat(u5, dim=1)

# Inputs to the model
x1 = torch.randn(1, 16777215)
x2 = torch.randn(1, 16777215)
x3 = torch.randn(1, 16777172)
# It should has more than one input tensor!
