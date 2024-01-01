 -- Add a non-identity operator for fun
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 64
        self.seq_len = 128
        self.dim = 65536
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.2, True)
        output = attn_weight @ value
        output = output + torch.randn(1, 64, 128, 65536)
        return output
import numpy as np

def generate_model_data(N, C_in, C_out, dtype=torch.float32):
    D = np.random.randint(65536, 4194304, size=(N, C_in, C_out))
    D = D.astype(dtype)
    D_torch = torch.tensor(D)
    scale = np.random.randint(1, 10, size=(N))
    scale = scale.astype(dtype)
    scale_torch = torch.tensor(scale)
    E = np.random.randint(1, 65536, size=(N, C_out, C_out))
    E = E.astype(dtype)
    E_torch = torch.tensor(E)
    return D_torch, D_torch * scale_torch, E_torch

batch_size = 1
num_heads = 128
seq_length = 2048
key_dim = 16384

D, D_scaled, E = generate_model_data(
    batch_size, key_dim, key_dim)
attn_mask = torch.randint(0, int(math.sqrt(seq_length)),
                          [batch_size, num_heads, seq_length, seq_length]
                          )
attn_mask = attn_mask.transpose(2, 3) - attn_mask
attn_mask.requires_grad_(False)

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.weight = nn.Parameter(torch.zeros((key_dim, key_dim)))
    self.bias = nn.Parameter(torch.ones((key_dim,)))
  def forward(self, query, key, value, mask):
    scaled_key = torch.softmax(E @ (self.weight + self.bias), dim=-1)
    return torch.bmm(scaled_key,
                    torch.bmm(torch.bmm(query, key), value))

input = torch.randn(2, num_heads, seq_length, key_dim)
out_pytorch = Model()(input, D_scaled, D, attn_mask)
print(out_pytorch)
# Output shape: (1, 128, 2048, 16384)
