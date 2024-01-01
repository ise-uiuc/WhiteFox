
class Model(torch.nn.Module):
    def __init__(self, *, d_model=256, nhead=8, dropout=0.2):
        super(Model, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(d_model=d_model, num_heads=nhead, dropout=dropout)
 
    def forward(self, x1, x2):
        y1, y2 = self.multihead_attention(x1, x1, x2)
        return y1, y2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 6, 15)
x2 = torch.randn(2, 14, 15)
