
B, T, H, W = 3, 3, 8, 8
num_heads = 4

class Model(torch.nn.Module):

  def __init__(self):
      super().__init__()
      self.attn = torch.nn.MultiheadAttention(H, num_heads)
  
  def forward(self, x1, x2, x3):
      v1 = self.attn(x1, x2, x2)
      v2 = self.attn(v1[1], x3, x3)
      return v2[0]

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(B, T, H)
x2 = torch.randn(B, T, H)
x3 = torch.randn(B, T, H)
