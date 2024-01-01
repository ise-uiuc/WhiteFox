
class Model(torch.nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, head_num, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
        self.dropout = torch.nn.Dropout(dropout_p)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.matmul1 = torch.nn.Linear(query_dim, head_num * key_dim)
        self.matmul2 = torch.nn.Linear(key_dim, head_num * value_dim)
 
    def forward(self, query, key, value):
        q = self.matmul1(query).reshape(-1, query.shape[1], self.head_num, self.key_dim)
        k = self.matmul1(key).reshape(-1, key.shape[1], self.head_num, self.key_dim)
        v = self.matmul1(value).reshape(-1, value.shape[1], self.head_num, self.key_dim)
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = math.sqrt(key.shape[-1])
        qk = qk / inv_scale_factor
        softmax_qk = self.softmax(qk)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, v)
        return output.reshape(-1, value.shape[1] * self.head_num)

m = Model(query_dim=64, key_dim=64, value_dim=32, head_num=4, dropout_p=0.7)
query = torch.randn(1, 8, 64)
key = torch.randn(1, 8, 64)
value = torch.randn(1, 8, 32)
output = m(query, key, value)
