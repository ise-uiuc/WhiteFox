
class Model(torch.nn.Module):
    def __init__(self, n_head, n_query, n_key, n_value, dropout_p=0.3):
        super().__init__()
        assert n_query % n_head == 0
        self.n_head = n_head
        self.n_q, self.n_k, self.n_v = n_query, n_key. n_value
        self.w_q = torch.nn.Linear(n_q, n_q, bias=False)
        self.w_k = torch.nn.Linear(n_k, n_k, bias=False)
        self.w_v = torch.nn.Linear(n_v, n_v, bias=False)
        self.fc2 = torch.nn.Linear(n_v, n_v, bias=False)
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, xq, xk, xv):
        q = self.w_q(xq)
        k = self.w_k(xk)
        v = self.w_v(xv)
        splitq = self.split(q, self.n_head)
        splitk = self.split(k, self.n_head)
        splitv = self.split(v, self.n_head)
        out, weights = [], []
        for qk, vk in zip(splitq, splitv):
            qktvk = self.attention(qk, vk)
            out.append(qktvk)
            weights.append(torch.triu(qktvk.masked_fill(qktvk.bool(), -1e9)))
        concated = torch.cat((torch.stack(out, dim=0)), dim=-1)
        return self.dropout(self.fc2(concated))

    def split(self, x, split_size):
        splitted = torch.split(x, split_size, dim=-1)
        stacked = [t.contiguous().view(x.size(0), -1, self.n_head, x.size(-1) // self.n_head) for t in splitted]
        return torch.stack(stacked, dim=-1)
    
    def attention(self, q, v):
        qktv = torch.matmul(q, torch.transpose(q, -2, -1))
        scaled_qktv = torch.mul(qktv, 1 / math.sqrt(qktv.size(-1)))
        softmax_qktv = scaled_qktv.softmax(dim=-1)
        dropout_qktv = torch.nn.functional.dropout(softmax_qktv, q=self.dropout_p)
        return torch.matmul(dropout_qktv, v)

# Initializing the model
n_head = 8
n_query = 16
n_key = 16
n_value = 16
