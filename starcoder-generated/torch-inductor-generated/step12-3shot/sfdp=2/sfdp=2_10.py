
class Model1(torch.nn.Module):
    def __init__(self, d_model=768, nhead=12, dropout_p=0.3):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.attn_dropout = dropout_p
        self.q_linear = torch.nn.Linear(d_model, d_model)
        self.k_linear = torch.nn.Linear(d_model, d_model)
        self.v_linear = torch.nn.Linear(d_model, d_model)
        self.out_linear = torch.nn.Linear(d_model, d_model)
 
    def forward(self, query, key, value, inv_scale_factor):
        bs, num, _ = key.size()
        q = self.q_linear(query).view(bs, num, self.nhead, self.d_model // self.nhead).permute(2, 0, 1, 3)
        k = self.k_linear(key).view(bs, num, self.nhead, self.d_model // self.nhead).permute(2, 0, 1, 3)
        v = self.v_linear(value).view(bs, num, self.nhead, self.d_model // self.nhead).permute(2, 0, 1, 3)
        q /= inv_scale_factor
        q = q.softmax(dim=-1)
        q = torch.nn.functional.dropout(q, self.attn_dropout)
        o = torch.matmul(q, v).permute(0, 2, 1, 3).contiguous().view(bs, num, self.d_model)
        return self.out_linear(o)

# Initializing the model
m = Model1(d_model=768)

# Inputs to the model
query = torch.randn(5, 86, 768)
key = torch.randn(5, 128, 768)
value = torch.randn(5, 128, 768)
inv_scale_factor = torch.tensor(1.0 / math.sqrt(86)).expand(5, 128, 128)
