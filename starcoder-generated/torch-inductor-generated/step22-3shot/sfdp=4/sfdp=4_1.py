
class Attention(torch.nn.Module):
    def __init__(self, n_head=1, d_model=512, d_head=64, score_type="dot_product"):
        super().__init__()
        assert score_type in ["dot_product"], "The specified score type '" + score_type + "' is not supported."
        self.n_head = n_head
        self.split_size = d_model // n_head
        self.scale = self.split_size ** -0.5
        self.linears = torch.nn.ModuleList()
        self.linears.append(torch.nn.Linear(d_model, d_head * n_head, bias=None))
        self.linears.append(torch.nn.Linear(d_head * n_head, d_model, bias=None))
 
    def forward(self, q, k, v, segment):
        qkv_same = q.data_ptr() == k.data_ptr() == v.data_ptr()
        kv_same = k.data_ptr() == v.data_ptr()
        qk_same = q.data_ptr() == k.data_ptr()
        # Linear projection
        if kv_same:
            qk = torch.einsum("binh,bmhn->bnm", torch.cat([q, q], dim=0), self.linears[0].weight)
            kqv = torch.einsum("binh,bmhn->bnm", k, self.linears[0].weight)
        elif qkv_same:
            qk = torch.einsum("binh,ohdn->bmno", torch.cat([q, q], dim=0), self.linears[0].weight)
            kqv = k
        elif qk_same:
            qk = torch.einsum("binh,ohdn->bmno", q, self.linears[0].weight)
            kqv = torch.einsum("binh,ohdn->bmno", k, self.linears[0].weight)
        else:
            qk = torch.einsum("binh,ohdn->bmno", q, self.linears[0].weight)
            kqv = torch.einsum("bmin,ohdn->bmno", k, self.linears[0].weight)
        # Scale
        qk = qk * self.scale
        # Split head or transform
        if qkv_same:
            qk = torch.split(qk, self.split_size, -1)
            kqv = torch.split(kqv, self.split_size, -1)
        elif qk_same:
            qk = torch.split(qk, self.split_size, 2)
        else:
            qk = torch.einsum("bmno->bmoi", qk).contiguous()
            kqv = torch.einsum("bmno->bmoi", kqv).contiguous()
        # Combine
        o = []
        for i_head in range(self.n_head):
            o.append(torch.einsum("bmoi,bmnm->bin", qk[i_head], kqv[i_head]))
        o = torch.cat(o, 2)
        # Transform
        o = self.linears[1](o)
        return o

# Initializing the model
attn = Attention(n_head=8)

# Inputs to the model
query = torch.randn(1, 64, 80)
key = torch.randn(1, 40, 80)
value = torch.randn(1, 40, 45)
