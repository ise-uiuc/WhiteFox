 initialization
d_model = 512
nhead = 4
dropout_p = 0.1
init_range = 0.02

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.multi_head_attention_layer = torch.nn.MultiheadAttention(d_model=d_model, num_heads=nhead)
 
    def forward(self, x1, x1, mask):
        out = self.multi_head_attention_layer(query=x1, key=x1, value=x1, key_padding_mask=mask)
        return out

# Inputs to the model
x1 = torch.rand((1, 8, 512))
x2 = torch.rand((1, 8, 512))
mask = torch.zeros((1, 8), dtype=torch.bool)
mask[:, 0] = True
