
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, heads, dropout_p):
        super().__init__()
 
        self.heads = heads
        self.d_model = d_model
 
        self.query = Linear(d_model, d_model)
        self.value = Linear(d_model, d_model)
        self.key = Linear(d_model, d_model)
 
        self.dropout = torch.nn.Dropout(dropout_p)
        self.output_linear = Linear(d_model, d_model)
 
    def forward(self, query, value, key=None, mask=None): # Here is the key argument
        attention = torch.matmul(query, key.transpose(-2, -1))
 
        inv_scale_factor = 1.0 / (self.d_model ** 0.5)
        scaled_attention = inv_scale_factor * attention
        if mask is not None:
            scaled_attention = scaled_attention.masked_fill(mask == 0, -1e9) # Applying mask
        softmax_attention = scaled_attention.softmax(dim=-1)
 
        output_attention = self.dropout(softmax_attention)
 
        output = torch.matmul(output_attention, value)
 
        return self.output_linear(output)

class Model(torch.nn.Module):
    def __init__(self, d_model, heads, dropout_p):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, heads, dropout_p)
 
    def forward(self, x1, x2, x3, mask1, mask2, mask3):
        t1 = self.attention(x3, x2, x3, mask3)
        t2 = self.attention(t1, x1, x1, mask1)
        v3 = self.attention(t1, x2, x2, mask2)
        return v3

# Initializing the model
m = Model(512, 8, 0.1)

# Inputs to the model
x1 = torch.randn(8, 64, 512)
x2 = torch.randn(8, 4, 512)
x3 = torch.randn(8, 20, 512)
m(__output__, __output__, __output__, __output__, __output__, __output__)

