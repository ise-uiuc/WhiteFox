
class Model(torch.nn.Module):
    def __init__(self, d_model=512, num_heads=8, dropout_p=0):
        super().__init__()
        self.d_model = d_model
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.to_qkv = torch.nn.Linear(d_model, d_model * 3)
    
    def forward(self, x1, x2, attn_mask):
        q, k, v = torch.chunk(self.to_qkv(x1), 3, dim=-1)
        q *= self.d_model ** -0.5
        qk = q @ k.transpose(-2, -1)
        qk += attn_mask
        qk = self.softmax(qk)
        qk = self.dropout(qk)
        o = qk @ v
        return o

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 128, 512)
x2 = torch.randn(2, 1000, 512)
attn_mask = torch.randn(3, 128, 1000)
