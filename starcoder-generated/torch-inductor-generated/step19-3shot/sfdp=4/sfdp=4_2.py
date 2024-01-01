
class Model(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_key, d_value, d_inner_hid, num_layers, dropout):
        super().__init__()
        self.slf_attn_mask = None
        self.slf_attn_layer = nn.ModuleList()
        for i in range(num_layers):
            self.slf_attn_layer.append(MultiHeadAttention(d_model, num_heads, d_key, d_value, dropout))
 
    def set_attn_mask(self, attn_mask):
        self.slf_attn_mask = attn_mask
 
    def forward(self, x1, x, x2):
        for attn_layer in self.slf_attn_layer:
            x = attn_layer(x1, x, x, attn_mask=self.slf_attn_mask)
        return x
 
# Initializing the model
m = Model(d_model=512,
          num_heads=8,
          d_key=64,
          d_value=64,
          d_inner_hid=1024,
          num_layers=4,
          dropout=0.1)
 
# Inputs to the model
x = torch.randn(5, 768, 512)
x1 = torch.randn(5, 768, 512)
x2 = x1 + x / x.sum(dim=0, keepdim=True)
