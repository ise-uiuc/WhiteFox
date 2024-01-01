
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        _v1 = v1.detach()
        inv_scale_factor = _v1.sum(-1, keepdim=True) ** 0.5
        v2 = v1.div(inv_scale_factor)
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.8)
        v5 = torch.matmul(v4, x2)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5, 64)
x2 = torch.randn(1, 6, 64)
