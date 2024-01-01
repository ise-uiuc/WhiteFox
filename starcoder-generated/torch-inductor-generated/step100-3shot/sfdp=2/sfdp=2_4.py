
class Model(torch.nn.Module):
    def __init__(self, 
                 dim=512, 
                 num_heads=8, 
                 dropout=0., 
                 bias=False):
        super().__init__()
        self.dim = dim
        self.qkv = torch.nn.Linear(dim, dim * 3, bias=bias)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.normalize_fact = dim ** -0.5
        self.proj = torch.nn.Linear(dim, dim)
        
    def forward(self, x1):
        qkv = self.qkv(x1).reshape(x1.size(0), 3, self.dim).transpose(-2, -1)
        q, k, v = qkv[...,:self.dim], qkv[...,self.dim:self.dim*2], qkv[...,self.dim*2:]
        q, k, v = [x.transpose(-2, -1) for x in (q, k, v)]
        scaled_qk = torch.matmul(q, k) * self.normalize_fact
        dropout_qk = self.dropout(torch.nn.functional.softmax(scaled_qk, dim=-1))
        output = torch.matmul(dropout_qk, v).transpose(-2, -1)
        return self.proj(output)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128, 512)
