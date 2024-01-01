
class Model(torch.nn.Module):
    def __init__(self, query, key, value, scale_factor, dropout_p):
        super().__init__()
        self.query = nn.Parameter(query)
        self.key = nn.Parameter(key)
        self.value = nn.Parameter(value)
        self.scale_factor = nn.Parameter(scale_factor)
        self.dropout_p = dropout_p
 
    def forward(self):
        qk = torch.matmul(self.query, self.key.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(self.value)
        return output

# Initializing the model
batch_size = 12
q_dim = 12
k_dim = 16
v_dim = 24
dim = 4
N = 10
query = torch.randn(batch_size, q_dim, dim, dim).to("cuda")
key = torch.randn(batch_size, N, k_dim, dim, dim).to("cuda")
value = torch.randn(batch_size, N, v_dim, dim, dim).to("cuda")
scale_factor = torch.tensor([1.0 / math.sqrt(dim)]).to("cuda")
dropout_p = 0.5
m = Model(query, key, value, scale_factor, dropout_p)

