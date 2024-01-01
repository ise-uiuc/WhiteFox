
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.1
        self.dropout = torch.nn.Dropout(self.dropout_p, inplace=True)
 
    def forward(self, x, mask):
        qkv = self.att(x)
        attn_weight = self.soft(qkv, mask)
        attn = self.dropout(attn_weight)
        output = self.proj(attn)
        return output
    
    def att(self, x):
        return x
    
    def soft(self, x, mask):
        return x
    
    def proj(self, x):
        return x

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(32, 8, 64)
mask = torch.ones(32, 8, 64)
