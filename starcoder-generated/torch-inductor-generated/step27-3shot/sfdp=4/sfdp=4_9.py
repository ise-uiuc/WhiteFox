
class Model(torch.nn.Module):
    def __init__(self, n_head, d_qkv):
        super().__init__()
        self.n_head = n_head
        assert d_qkv % n_head == 0
        self.d_head = d = d_qkv // n_head

    def forward(self, query, key, value, attn_mask=None):
        n_state, bsz, n_head, d_head = key.size(), key.size(1), self.n_head, self.d_head
        v1 = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(n_state)
        v2 = v1 + (attn_mask.unsqueeze(-3) if attn_mask is not None else 0)
        attn_weight = attn_weight = torch.softmax(v2.contiguous().view(-1, bsz * n_head, n_state).transpose(0, 1), 1)
        output = torch.matmul(attn_weight.view(bsz * n_head, -1, n_state), value)
        return output.view(bsz, n_head, d_head)

# Initializing the model
m = Model(8, 256)

# Inputs to the model
x1 = torch.rand(1, 3, 64, 64)
x2 = torch.rand(1, 3, 64, 64)
x3 = torch.rand(1, 3, 64, 64)
output = m(x1, x2, x3)

