
class MyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__() 
        self.normalized_shape = normalized_shape 
        self.weight = nn.Parameter(torch.ones(normalized_shape)) 
        self.bias = nn.Parameter(torch.zeros(normalized_shape))  
        self.variance_epsilon = eps 

    def forward(self, x): 
        u = x.mean(-1, keepdim=True) 
        s = (x - u).pow(2).mean(-1, keepdim=True) 
        x = (x - u) / torch.sqrt(s + self.variance_epsilon) 
        return self.weight * x + self.bias

# Model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = MyLayerNorm((8, 32, 32))
 
    def forward(self, x, weight):
        v1 = torch.matmul(x, weight)
        v2 = v1.transpose(-2, -1)
        v3 = v2/31
        v4 = torch.softmax(v3, dim=-1)
        v5 = torch.matmul(v4, x)
        return v5

# Initializing the model
m = Model()
weight = torch.randn(8, 32, 32)
x = torch.randn(1, 8, 32, 32)
