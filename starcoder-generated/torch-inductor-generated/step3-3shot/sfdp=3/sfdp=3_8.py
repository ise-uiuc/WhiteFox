
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_head, d_model, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.dropout = dropout
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(n_head, n_head, bias=False)
 
    def get_d_head(self):
        return self.d_model // self.n_head
 
    def forward(self, q, k, v):
        q = self.q_linear(q).view(nq, d_head, self.n_head)
        k = self.k_linear(k).view(nk, d_head, self.n_head)
        v = self.v_linear(v).view(nk, d_head, self.n_head)
        q, k, v = [x.transpose(1, 2) for x in [q, k, v]]
        a = torch.matmul(q, k) / math.sqrt(d_head)
        b = self.drop(F.softmax(a, dim=-1))
        c = torch.matmul(b, v)
        d = d_head * self.dropout(c)
        output = d.view(nq, d_all)
        output = self.fc(output)
        return output
 
class Model(nn.Module):
    def __init__(self, n_head, d_model, hidden_size, dropout_p):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.mha = MultiHeadAttention(self.n_head, self.d_model)
        self.drop = nn.Dropout(dropout_p)
        self.fc = nn.Linear(d_model, d_model)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
 
    def forward(self, x, mask):
        attn = self.mha(x, x, x)
        attn = self.drop(attn)
        attn_output = self.fc(attn) + x
        return attn

# Initializing the model
m = Model(n_head=n_head, d_model=d_model, hidden_size=hidden_size, 
          dropout_p=dropout_p)

# Inputs to the model
x = torch.randn(nq, d_all) # query input tensor
mask = torch.randn(nq, nk) # masking tensor
