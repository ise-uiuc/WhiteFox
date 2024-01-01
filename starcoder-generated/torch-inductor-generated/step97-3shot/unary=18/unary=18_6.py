
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.sigmoid(x1)
        v2 = v1.unsqueeze(1)
        v3 = v2.expand([64,3,64,64])
        v4 = v3.permute([0,2,3,1])
        return v4

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
