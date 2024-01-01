
class SelfAttention(torch.nn.Module):
  def __init__(self, embed_dim, num_heads, dropout_p, attention_type='dot'):
    super().__init__()
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.dropout_p = dropout_p
    self.attention_type = attention_type

    self.fc1 = torch.nn.Linear(embed_dim, embed_dim)
    self.fc2 = torch.nn.Linear(embed_dim, embed_dim)
    self.attn_dropout = torch.nn.Dropout(dropout_p)

    if attention_type == 'general':
      self.attn = torch.nn.Linear(embed_dim, embed_dim)
    else: 
      self.attn = None

  def forward(self, x):
    query = self.fc1(x)
    value = self.fc2(x)
    
    attn1 = torch.matmul(query, value.transpose(-2, -1))
    if self.attention_type == 'general':
      attn2 = self.attn(x)
      attn = attn1 + attn2
    else:
      attn = attn1
    attn = self.attn_dropout(attn.softmax(dim=-1))
    return torch.matmul(attn, value)
layer = SelfAttention(embed_dim=1024, num_heads=4, dropout_p=0.1, attention_type='dot')

# Initializing the model
torch.manual_seed(0)
x = torch.randn(100, 16, 1024)
