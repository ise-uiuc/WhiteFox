
class Model(torch.nn.Module):
    def __init__(self, head_num, size_per_head):
        super(Model, self).__init__()
        self.head_num = head_num
        self.head_dim = size_per_head
        self.all_head_dim = self.head_num * self.head_dim
        self.linear_q = torch.nn.Linear(5, self.all_head_dim)
        self.linear_k = torch.nn.Linear(4, self.all_head_dim)
        self.linear_v = torch.nn.Linear(6, self.all_head_dim)
        self.linear_merge = torch.nn.Linear(self.all_head_dim, 3)
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)
        q = q.reshape(-1, self.head_num, self.head_dim)
        k = k.reshape(-1, self.head_num, self.head_dim)
        v = v.reshape(-1, self.head_num, self.head_dim)
        q = q.transpose(-2, -1)
        qk = torch.matmul(q, k)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = torch.nn.functional.softmax(scaled_qk, dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        dropout_qk = dropout_qk.matmul(v)
        dropout_qk = dropout_qk.transpose(-2, -1)
        dropout_qk = dropout_qk.reshape(-1, self.all_head_dim)
        output = self.linear_merge(dropout_qk)
        return output

# Initializing the model
m = Model(8, 16)

# Inputs to the model
query = torch.randn(6, 5, 8)
key = torch.randn(4, 6, 16)
value = torch.randn(8, 4, 24)
inv_scale_factor = torch.randn(1, 8, 64)
dropout_p = 0.1
