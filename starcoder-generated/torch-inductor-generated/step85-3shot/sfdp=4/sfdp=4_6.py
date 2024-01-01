
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs, self_attn_mask, encoder_attn_mask):
        queries = self.q_lin(inputs)
        keys = self.k_lin(inputs)
        vals = self.v_lin(inputs)
        # Self-attention
        q, k, v, mask = queries, keys, vals, self_attn_mask
        out = self.sdp(q, k, mask).to(inputs.device)
        output = self.out(out).to(inputs.device)
        # Cross-Attention
        q, k, v, mask = self.q_tran(queries), self.k_tran(keys), self.v_tran(vals), encoder_attn_mask
        out = self.sdpxv(q, k, mask).to(inputs.device)
        output = self.out(out).to(inputs.device)
        return output
# Inputs to the model
inputs = torch.randn(2, 10, 32, 32)
self_attn_mask = (torch.rand(1, 32, 32) > 0.5).fill_(-1000000000.0)
enc_attn_mask = (torch.rand(1, 32, 32) > 0.5).fill_(-1000000000.0)
