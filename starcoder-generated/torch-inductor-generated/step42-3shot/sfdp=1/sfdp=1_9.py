
class Model(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(d_model, nhead)
 
    def forward(self, src, src_mask=None):
        output, _ = self.attention(src, src, src, src_mask=src_mask)
        return output

# Initializing the model
m = Model()

# Inputs to the model
src = torch.randn(4, 32, 128)
src_mask = torch.randn(4, 4)
