
class Model(torch.nn.Module):
    def __init__(self, dim, shape):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(dim, *shape))
        self.key = torch.nn.Parameter(torch.randn(dim, *shape))
        self.value = torch.nn.Parameter(torch.randn(dim, *shape))
        self.inv_scale_factor = torch.nn.Parameter(torch.randn(*shape))
 
    def forward(self, query):
        q = self.query.view(1, -1, 1, 1, 1)
        k = self.key.view(1, 1, -1, 1, 1)
        v = self.value.view(1, 1, 1, -1, 1)
        inv_scale_factor = self.inv_scale_factor.view(1, 1, 1, 1, -1)
        q = q.expand(-1, -1, *query.size(-3:-2), -1)
        k = k.expand(-1, *query.size(-2:-1), -1, -1)
        v = v.expand(-1, *query.size(-2:-1), -1, -1)
        inv_scale_factor = inv_scale_factor.expand(-1, *query.size(-2:-1), -1, -1)        
        q = q.squeeze()
        k = k.squeeze()
        v = v.squeeze()
        inv_scale_factor = inv_scale_factor.squeeze()        
        