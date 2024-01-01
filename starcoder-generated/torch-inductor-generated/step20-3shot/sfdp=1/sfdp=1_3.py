
class Model(torch.nn.Module):
    def __init__(self, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.w_q = torch.nn.Linear(16, 16 * num_heads, bias=False)
        self.w_k = torch.nn.Linear(24, 16 * num_heads, bias=False)
        self.w_v = torch.nn.Linear(32, 16 * num_heads, bias=False)
        self.projection = torch.nn.Linear(16 * num_heads, 16)
 
    def forward(self, q, k, v, inv_scale_factor, dropout_p):
        q = torch.stack([self.w_q(q) for _ in range(self.num_heads)], dim=1)
        k = torch.stack([self.w_k(k) for _ in range(self.num_heads)], dim=1)
        v = torch.stack([self.w_v(v) for _ in range(self.num_heads)], dim=1)
        qk = torch.matmul(q.transpose(-2, -1), k) # (batch, num_heads, tgt_len, src_len)
        scaled_qk = qk.div(inv_scale_factor) # (batch, num_heads, tgt_len, src_len)
        softmax_qk = scaled_qk.softmax(dim=-1) # (batch, num_heads, tgt_len, src_len)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p) # (batch, num_heads, tgt_len, src_len)
        output = dropout_qk.matmul(v) # (batch, num_heads, tgt_len, head_dim)
        output = output.transpose(1, 2) # (batch, tgt_len, num_heads, head_dim)
        first_head = output[:, :, 0, :] # (batch, tgt_len, head_dim)
        logits = self.projection(first_head) # (batch, tgt_len, num_heads)
        return logits

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 1, 16)
k = torch.randn(1, 4, 24)
v = torch.randn(1, 2, 32)
inv_scale_factor = torch.tensor([2.0])
