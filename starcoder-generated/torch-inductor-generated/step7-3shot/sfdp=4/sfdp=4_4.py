
class Model(torch.nn.Module):
    def __init__(self, key_dim, query_dim, n_head, n_hidden):
        super().__init__()
        self.key_dim = key_dim
        self.query_dim = query_dim
        self.n_head = n_head
        self.n_hidden = n_hidden
        self.w_q = torch.nn.Linear(query_dim, n_hidden * n_head, bias=True)
        self.w_k = torch.nn.Linear(key_dim, n_hidden * n_head, bias=False)
        self.w_v = torch.nn.Linear(n_hidden, n_hidden * n_head, bias=False)
        self.proj = torch.nn.Linear(n_hidden * n_head, n_hidden)
 
    def forward(self, inputs):
        q = self.trans_q(inputs)
        k = self.trans_k(inputs)
        v = self.trans_v(inputs)
        attn_w = torch.softmax(torch.matmul(q, torch.transpose(k, 2, 3)) / np.sqrt(self.key_dim), dim=-1)
        outputs = torch.matmul(attn_w, v)
        outputs = torch.reshape(outputs, [-1, self.n_head * self.n_hidden])
        outputs = self.proj(outputs)
        return outputs
 
    def trans_q(self, inputs):
        q = self.w_q(inputs)
        q = torch.reshape(q, [-1, self.n_hidden, self.n_head])
        return q
 
    def trans_k(self, inputs):
        k = self.w_k(inputs)
        k = torch.reshape(k, [-1, self.n_hidden, self.n_head])
        return k
 
    def trans_v(self, inputs):
        v = self.w_v(inputs)
        v = torch.reshape(v, [-1, self.n_head, self.n_hidden])
        return v

# Initializing the model
m = Model(key_dim=3, query_dim=3, n_head=8, n_hidden=16)

# Inputs to the model
inputs = torch.randn(5, 10, 3)
