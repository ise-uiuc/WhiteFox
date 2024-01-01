
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = F.conv2d(x1, x2, None)
        inv_scale_factor = 1 / (x1.shape[-1]*x1.shape[-2] + x2.shape[-1]*x2.shape[-2])
        v2 = v1.div(inv_scale_factor)
        v3 = F.softmax(v2, dim=-1)
        v4 = F.dropout(v3,.5, True, False)
        v5 = F.conv2d(v4, x2, None)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)

