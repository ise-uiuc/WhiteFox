
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_linear = torch.nn.Linear(512, 256)
        self.k_linear = torch.nn.Linear(512, 256)
    
    def self_attention(self, q, k, v, s, dropout):
        q = self.q_linear(q)
        k = self.k_linear(k)
        n = k.size(-2)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(s)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout)
        output = dropout_qk.matmul(v)
        return output
    
    def forward(self, query, key, value, scale_factor, dropout_p):
        v1 = self.self_attention(query, key, value, scale_factor, dropout_p)
        return v1

# initial model
m = Model()
# Initializing the model
query = torch.randn(1, 8, 256)
key = torch.randn(1, 8, 256)
value = torch.randn(1, 8, 256)
scale_factor = torch.randn(1, 1, 1, 1) + 1
dropout_p = torch.empty((1,)).uniform_()
