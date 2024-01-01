
class Model(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = torch.nn.Linear(hidden_dim, hidden_dim)
        self.key = torch.nn.Linear(hidden_dim, hidden_dim)
        self.value = torch.nn.Linear(hidden_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(0.1)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, q, k, v, inv_scale_factor):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, v)
        return output

# Initializing the model
m = Model(32)

# Inputs to the model
q = torch.randn(2, 8, 32)
k = torch.randn(2, 4, 32)
v = torch.randn(2, 4, 32)
inv_scale_factor = torch.tensor(float(math.sqrt(1.0 / 4)))
