
class Model(torch.nn.Module):
  def __init__(self, in_dim, head, n_layers, hidden_dim, dropout_p=0.1):
    super().__init__()
    self.n_layers = n_layers
    self.heads = head
    self.hidden_dim = hidden_dim
    self.dropout_p = dropout_p
    self.lin_q = torch.nn.Linear(in_dim, in_dim)
    self.lin_k = torch.nn.Linear(in_dim, in_dim)
    self.lin_v = torch.nn.Linear(in_dim, in_dim)
    self.lin_o = torch.nn.Linear(in_dim, in_dim)
    self.lin_o_2 = torch.nn.Linear(in_dim, in_dim)

  def forward(self, q, k, v, attn_mask):
    bs, head, d_hid, d_head = q.size(0), self.heads, self.hidden_dim, self.hidden_dim//self.heads
    q = self.lin_q(q).view(bs, d_hid, head, d_head).transpose(1, 2)
    k = self.lin_k(k).view(bs, d_hid, head, d_head).transpose(1, 2)
    v = self.lin_v(v).view(bs, d_hid, head, d_head).transpose(1, 2)
    attn_mask = attn_mask.view(bs * head, 1, 1, d_hid)

    qk = q @ k.transpose(-2, -1) / math.sqrt(d_hid)
    qk = qk + attn_mask
    attn_weight = torch.softmax(qk, dim=-1)
    attn_weight = torch.dropout(attn_weight, p=self.dropout_p, training=True)
    output = attn_weight @ v
    output = output.transpose(1, 2).contiguous().view(bs, -1, d_hid*head)
    output = self.lin_o(output)

    return output

# Initializing the model
m = Model(1024, 8, 6, 1024)

# Inputs to the model
x1 = torch.randn(768, 1024)
x2 = torch.randn(768, 1024)
x3 = torch.randn(768, 1024)
x4 = torch.zeros(768, 1024)
