
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = torch.nn.Linear(64, 64)
 
    def forward(self, query, key, value, dropout_p, inv_scale_factor):
        v_qk = self.qkv(query)
        qk = v_qk.matmal(key.transpse(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(32, 1, 64)
key = torch.randn(32, 1, 64)
value = torch.randn(32, 1, 64)
dropout_p = 0.1
inv_scale_factor = 0.1
