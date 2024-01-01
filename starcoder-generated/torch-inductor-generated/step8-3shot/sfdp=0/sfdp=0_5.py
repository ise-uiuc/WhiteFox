
class Model(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
 
    def forward(self, q, k, v, scale, mask):
        scale = scale.squeeze()
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        inv_scale = scale.pow(-1)
        inv_scale = inv_scale.view(1, self.dim, 1, 1)
        __output__, 