
class Model(torch.nn.Module):
    def __init__(self, dim_hidden):
        super().__init__()
        self.w1 = torch.nn.Linear(dim_hidden, dim_hidden)
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.w2 = torch.nn.Linear(dim_hidden, dim_hidden)
 
    def forward(self, x1, x2):
        q1, k1, v1 = x1, x2, x2
        q2 = self.w1(q1)
        k2 = self.w1(k1)
        v2 = self.w2(v1)
 
        q2 = q2 * attn_mask.transpose(-1, -2)
        k2 = k2 * attn_mask.transpose(-1, -2)
        Q = torch.matmul(q2, k2.transpose(-2, -1))
        H = Q / scale_factor
        H = H - torch.max(Q, dim=-1, keepdim=True)[0]
 
        A = torch.nn.functional.softmax(H, dim=-1)
        B = self.dropout(A)
        C = torch.matmul(B, v2)
        D = C + q1
        return D

# Initializing the model
m = Model(dim_hidden)

# Inputs to the model
x1 = torch.randn(10, 19, dim_hidden) # (seq_len, batch_size, dim_hidden)
x2 = torch.randn(10, 19, dim_hidden) # (seq_len, batch_size, dim_hidden)
attn_mask = torch.zeros((10, 1, 19))
