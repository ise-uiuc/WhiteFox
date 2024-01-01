
class Model(torch.nn.Module):
    def __init__(self, num_heads: int, hidden_dim: int):
        super().__init__()
        self.query = torch.nn.Linear(hidden_dim, hidden_dim)
        self.key = torch.nn.Linear(hidden_dim, hidden_dim)
        self.value = torch.nn.Linear(hidden_dim, hidden_dim)
 
    def forward(self, x1, x2):
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(0.06)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1, training=True)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model(4, 512)

# Inputs to the model
x1 = torch.randn(1, 768)
x2 = torch.randn(1, 768)
