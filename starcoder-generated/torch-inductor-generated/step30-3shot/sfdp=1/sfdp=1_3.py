
class Model(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.w_q = torch.nn.Linear(hidden_size, hidden_size)
        self.w_k = torch.nn.Linear(hidden_size, hidden_size)
        self.w_v = torch.nn.Linear(hidden_size, hidden_size)
        self.w_o = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(p=dropout_p)
 
    def forward(self, q, k, v, mask):
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, v)
        return output

# Initializing the model
hidden_size = 10
n = 4
m = Model(hidden_size)
query = torch.randn(n, 1, hidden_size)
key = torch.randn(n, 10, hidden_size)
value = torch.randn(n, 10, hidden_size)
mask = torch.tril(torch.ones((n, 1, 10)))
