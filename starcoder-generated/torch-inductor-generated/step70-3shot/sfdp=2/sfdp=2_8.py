
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(64, 4, 0.10000000149011612)
        self.dropout2 = torch.nn.Dropout(0.10000000149011612)
 
    def forward(self, x1):
        v1, v2, v3 = self.attn(x1, x1, x1, need_weights=False)
        v4 = self.dropout2(v2)
        v5 = v4.matmul(v3)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8, 64, 256)
