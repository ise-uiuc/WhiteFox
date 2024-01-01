
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v0 = x1.permute(0, 2, 1)
        v1 = x2.permute(0, 2, 1)
        v2 = torch.bmm(v0, v1)
        v2 = v2.reshape(12)
        #return v2
        return torch.matmul(v2.reshape([3,4]),v2.reshape([4,3]))
# Inputs to the model
x1 = torch.randn(1, 3, 4)
x2 = torch.randn(1, 4, 3)
