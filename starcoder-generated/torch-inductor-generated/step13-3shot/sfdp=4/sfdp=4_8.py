
class Model(torch.nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, d_ff):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_ff = d_ff
        self.w_qs = torch.nn.Linear(d_model, n_head * d_k)
        self.w_ks = torch.nn.Linear(d_model, n_head * d_k)
        self.w_vs = torch.nn.Linear(d_model, n_head * d_v)
        self.fc = torch.nn.Linear(n_head * d_v, d_model)
 
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = query.size(0), query.size(1), key.size(1), value.size(1)
        query = self.w_qs(query).view(sz_b, len_q, n_head, d_k)
        key = self.w_ks(key).view(sz_b, len_k, n_head, d_k)
        value = self.w_vs(value).view(sz_b, len_v, n_head, d_v)
        query, key, value = query.permute(2, 0, 1, 3), key.permute(2, 0, 1, 3), value.permute(2, 0, 1, 3)
        x = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        x = x + attention_mask
        x = torch.softmax(x, dim=-1)
        x = torch.matmul(x, value)
        x = x.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        x = self.fc(x)
        return x

# Initializing the model
model = Model(n_head=4, d_model=10, d_k=10, d_v=20, d_ff=20)

# Inputs to the model
q = torch.randn(5, 10, 10)
k = torch.randn(5, 15, 10) 
v = torch.randn(5, 15, 20)
attention_mask = torch.randn(5, 1, 15, 15).ge(1).type(torch.FloatTensor)
