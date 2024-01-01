
class Model(torch.nn.Module):
    def __init__(self, query_dim, key_dim):
        super().__init__()
        self.matmul_qk = torch.nn.Linear(query_dim, key_dim)
        self.matmal_v = torch.nn.Linear(query_dim, key_dim)
        self.softmax_qk = torch.nn.Softmax(dim=-1)
        self.dropout_qk = torch.nn.Dropout(p=0.2)
 
    def forward(self, query, key, value, scale_factor=1, dropout_p=0):
        qk = self.matmul_qk(query)
        qk_scaled = qk.div(scale_factor)
        softmax_qk = self.softmax_qk(qk_scaled)
        dropout_qk = self.dropout_qk(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(query_dim=3, key_dim=5)

# Inputs to the model
query = torch.randn(2, 6, 3)
key = torch.randn(2, 5, 3)
value = torch.randn(2, 5, 3)
