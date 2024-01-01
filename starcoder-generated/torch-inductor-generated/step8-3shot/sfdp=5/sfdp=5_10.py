
class Model(torch.nn.Module):
    def __init__(self, dropout_p=0.2):
        super().__init__()
        self.dropout_p = dropout_p
        self.scale = math.sqrt(self.dropout_p)
 
    def forward(self, q, k, v, attn_mask):
        attn = q @ k.transpose(-2, -1) / self.scale # Dot product
        attn = attn + attn_mask # Adding a mask
        attn = torch.softmax(attn, dim=-1)
        attn = torch.dropout(attn, self.dropout_p, True) # Applying a dropout
        attn = attn @ v
        return attn

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(2, 3, 4)
k = torch.randn(2, 4, 6)
v = torch.randn(2, 4, 8)
attn_mask = torch.tril(torch.ones(2, 3, 4)) # (batch_size, seq_len, seq_len)
