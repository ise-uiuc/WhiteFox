
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 64
        self.dropout_p = 0.2
 
    def forward(self, query, key, value, inverse_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inverse_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        res = dropout_qk.matmul(value)
        return res

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 3, 48, 64)
key = torch.randn(1, 4, 48, 64)
value = torch.randn(1, 4, 48, 64)
inverse_scale_factor = torch.randn(64, 64)
