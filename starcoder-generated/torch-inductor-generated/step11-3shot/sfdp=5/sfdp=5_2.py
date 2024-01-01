
class Transformer(nn.Module):
 def forward(self, query, key, value, attn_mask=None, dropout_p=0.1):
   qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
   if attn_mask:
     qk = qk + attn_mask
   attn_weight = torch.softmax(qk, dim=-1)
   attn_weight = torch.dropout(attn_weight, dropout_p, True)
   return attn_weight @ value

# Initializing the model
m = Transformer()

# Initialize query, key, value
query = torch.randn(2, 5, 10)
key = torch.randn(2, 5, 10)
value = torch.randn(2, 5, 10)

# Input of attention mask
attn_mask = torch.randn(5, 5)

