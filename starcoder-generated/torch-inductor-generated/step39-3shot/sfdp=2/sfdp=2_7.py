
class Model(torch.nn.Module):
    def __init__(self, dim, num_heads, batch_size, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
        self.softmax_qk = torch.nn.Softmax(dim=-1)
        self.dropout_qk = torch.nn.Dropout(self.dropout_p)
        self.matmul1 = torch.nn.Linear(dim, 1)
        self.matmul2 = torch.nn.Linear(dim, 1, bias=False)
 
    def forward(self, query, key, value, scale_factor, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = self.softmax_qk(scaled_qk)
        dropout_qk = self.dropout_qk(softmax_qk)
        return torch.matmul(dropout_qk, value), dropout_qk

# Initializing the model
dim = 64
num_heads = 2
batch_size = [2, 4]
dropout_p = 0.2
model = Model(dim=dim, num_heads=num_heads, batch_size=batch_size, dropout_p=dropout_p)

# Inputs to the model
query = torch.randn(batch_size, 2, dim)
key = torch.randn(batch_size, 4, dim)
value = torch.randn(batch_size, 4, dim)
scale_factor = 1. / num_heads  # or `1. / math.sqrt(dim)`
inv_scale_factor = 1. / dropout_p
dropout_p = 0.2
__output__, dropout_qk = model(query, key, value, scale_factor, inv_scale_factor, dropout_p)
