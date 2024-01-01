
class Model(torch.nn.Module):
    def __init__(self, d_model, nhead, dropout=0.2):
        super().__init__()
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.dropout = dropout
        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)
 
    def forward(self, query, key, value):
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
 
        q = q.view(q.size(0), q.size(1), self.nhead, self.d_k).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.nhead, self.d_k).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.nhead, self.d_k).transpose(1, 2)
 
        qk = torch.matmul(q, k.transpose(2, 3))
        qk = qk.div(math.sqrt(self.d_k))
        qk = qk.softmax(dim=-1)
        qk = torch.nn.functional.dropout(qk, p=self.dropout)

        output = torch.matmul(qk, v)
        output = output.transpose(1, 2).contiguous().view(output.size(0), output.size(1), output.size(2) * output.size(3))

        return output

# Initializing the model
model = Model(d_model=512, nhead=8)

# Inputs to the model
x1 = torch.randn(5, 60, 512)
x2 = torch.randn(5, 150, 512)
x3 = torch.randn(5, 150, 512)
