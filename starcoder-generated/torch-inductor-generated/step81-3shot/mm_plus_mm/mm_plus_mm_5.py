
class Model(nn.Module):
    def __init__(self, dim_w, dim_x):
        super(Model, self).__init__()
        self.dim_w = dim_w
        self.dim_x = dim_x
    def forward(self, w, x):
        y = w*w + x*x
        return y
# Inputs to the model
w = torch.randn(15, 15)
x = torch.randn(15, 15)
