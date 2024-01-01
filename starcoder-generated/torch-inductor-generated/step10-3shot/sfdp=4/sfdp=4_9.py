
class Model(torch.nn.Module):
    def __init__(self):
        self.head_num = 16
        self.size_per_head = 64

        self.qkv_weights = torch.nn.Linear( 48, 3 * self.head_num * self.size_per_head )
        self.out_weights = torch.nn.Linear( self.head_num * self.size_per_head, 1600 )

    def forward(self, l):
        sz = l.size()
        qkvw    = self.qkv_weights(l) # (batch_sz, 128, 3 * 16 * 64)
        qkv     = qkvw.view(sz[0], sz[1], 3, self.head_num, self.size_per_head)  #  (batch_sz, 128, 3, 16, 64)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2] # three tensors of (batch_sz, 128, 16, 64)

        d = q.size(-1)
        attn_mask = (torch.triu(torch.full((d, d), 1e9, device=x.device, dtype=x.dtype))
                     * torch.triu(torch.full((d, d), 0,    device=x.device, dtype=x.dtype), 1))
        attn_mask = attn_mask.unsqueeze(0).expand(sz[0], 1, d, d)

        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + attn_mask

        attn_weight = torch.softmax(qk, dim=-1)

        return (attn_weight @ v).view(sz) @ self.out_weights(q)

m = Model();

# Inputs to the model
x1 = torch.randn(1, 128, 48)
