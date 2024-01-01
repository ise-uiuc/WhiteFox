
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1.permute(0, 2, 1)
        x1_dim_1 = v1.shape[0]
        x1_dim_2 = v1.shape[1]
        v2 = x2.permute(0, 2, 1)
        x2_dim_0 = v2.shape[0]
        x2_dim_2 = v2.shape[2]
        v3 = torch.bmm(v1, v2)
        v4 = v3[0][0][0]
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
