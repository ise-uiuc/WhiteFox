
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.3
        self.scale_factor = 1 / np.sqrt(d_k)
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        outputs = dropout_qk.matmul(value)
        return outputs
 
# Initializing the model
m = Model()
 
# Inputs to the model
query = torch.randn(64, query_len, d_k)
key = torch.randn(64, key_len, d_k)
value = torch.randn(64, key_len, d_k)
__outputs__ = m(query, key, value)
