
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention()
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        output, _ = self.attn(query, key, value, attn_mask=None, 
                                key_padding_mask=None, need_weights=False, 
                                attn_mask=None, dropout_p=dropout_p)

        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(8, 6, 128)
key = torch.randn(8, 4, 128)
value = torch.randn(8, 4, 128)
inv_scale_factor = 1.0/(query.shape[-1]**0.5)
dropout_p = 0.5
