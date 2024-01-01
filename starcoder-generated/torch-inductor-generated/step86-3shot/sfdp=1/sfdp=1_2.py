
class Model(torch.nn.Module):
    def __init__(self, num_q, num_k, num_v, scale_factor, dropout_p):
        super().__init__()
        self.k = torch.nn.Linear(num_k, num_q)
        self.v = torch.nn.Linear(num_v, num_q)
        self.dropout_p = dropout_p
        self.scale_factor = scale_factor
 
    def forward(self, query, key, value):
        k = self.k(key).transpose(-2, -1)
        v = self.v(value).transpose(-2, -1)
        qk = torch.matmul(query, k)
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model(num_q=256, num_k=256, num_v=256, scale_factor=7 + 1e-6, dropout_p=0.0)

# Inputs to the model
query = torch.randn(2, 256, 64)
key = torch.randn(2, 256, 64)
value = torch.randn(2, 256, 64)
