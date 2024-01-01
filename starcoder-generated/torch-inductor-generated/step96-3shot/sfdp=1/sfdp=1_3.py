
class Model(torch.nn.Module):
    def __init__(self, d_model, n_head, dropout, bias=False):
        super().__init__()
        assert n_head!= 0
        assert d_model % n_head == 0
        self.scale_factor = math.sqrt(d_model)
        self.d_model = d_model
        self.n_head = n_head
        self.w_q = torch.nn.Linear(d_model, d_model, bias=bias)
        self.w_k = torch.nn.Linear(d_model, d_model, bias=bias)
        self.w_v = torch.nn.Linear(d_model, d_model, bias=bias)
        self.dropout_layer = torch.nn.Dropout(dropout)
 
    def forward(self, query, key, value, attn_mask=None):
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)
        attn_mask = attn_mask.repeat_interleave(self.n_head, dim=0)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout_layer(softmax_qk)
        return dropout_qk.matmul(v)

# Initializing the model
m = Model(d_model=64, n_head=4, dropout=0.1)

# Inputs to the model
query = torch.randn(1, 128, 64)
key = torch.randn(1, 128, 64)
value = torch.randn(1, 128, 64)
attn_mask = torch.ones(1, 64, 64).bool()
