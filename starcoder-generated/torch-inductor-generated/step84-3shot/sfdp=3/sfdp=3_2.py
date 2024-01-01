
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        out_dim = 512
        in_dim = 512
        emb_dim = 64
        self.scale_factor = 1 / math.sqrt(emb_dim)
        self.to_qkv = torch.nn.Linear(in_dim, 3 * out_dim, bias=False)
        self.to_out = torch.nn.Linear(out_dim, in_dim)
        self.dropout = torch.nn.Dropout(0.2)
 
    def forward(self, x1):
        v1 = x1.reshape(x1.size()[0], -1)
        v2 = self.to_qkv(v1).reshape(v1.size()[0], 3, 512).transpose(-2, -1)
        v3 = v2 * self.scale_factor
        v4 = v3.softmax(dim=-1)
        v5 = self.dropout(v4)
        return v5
    
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 80, 512)
