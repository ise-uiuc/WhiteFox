
class Model(torch.nn.Module):
    def __init__(self, num_heads, embed_size, dropout_p=0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_p)
        self.dropout_p = dropout_p
 
    def forward(self, enc_input, dec_input):
        q = torch.matmul(query, self.key.transpose(-2, -1))
        k = torch.matmul(value, self.key.transpose(-2, -1))
        v = torch.matmul(key, self.value.transpose(-2, -1))
        q /= self.scale_factor
        dot_product = q @ k.transpose(-2, -1)
        softmax_attn = nn.functional.softmax(dot_product, dim=-1).type_as(query)
        attn = self.dropout(attn)
        out = attn @ v
        return out

# Initializing the model
m = Model(3, 6)

# Inputs to the model
enc_input = torch.randn(2, 3, 5)
dec_input = torch.randn(2, 3, 4)
