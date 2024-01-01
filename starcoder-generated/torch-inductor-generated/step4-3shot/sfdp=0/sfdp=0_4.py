
class Model(torch.nn.Module):
    def __init__(self, dim=768):
        self.qkv = torch.nn.Linear(dim, dim * 3)
        self.out = torch.nn.Linear(dim, dim)
        self.qkv.apply(self._init_weights)
        self.out.apply(self._init_weights)
 
        self.dim = d
 
    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear)):
            torch.nn.init.xavier_uniform(module.weight)
            if isinstance(module, (torch.nn.Linear)) and module.bias is not None:
                constant_(module.bias, 0.0)
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0.0)
            torch.nn.init.constant_(module.weight, 1.0)
 
    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q /= math.sqrt(self.dim)
        attention = torch.matmul(q, k.transpose(-2, -1))
        attention = torch.softmax(attention, dim=-1)
        out = torch.matmul(attention, v)
        out = self.out(out)
        return out

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 768)
y = m(x1)

