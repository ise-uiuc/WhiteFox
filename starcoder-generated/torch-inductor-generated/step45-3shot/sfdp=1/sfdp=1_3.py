
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 16
        self.head_dim = 128
        self.scale_factor = np.sqrt(self.head_dim)
 
    def forward(self, query, key, value):
        q = query
        k = key
        v = value
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.2)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 4, 128, 128)
key = torch.randn(1, 4, 128, 128)
value = torch.randn(1, 4, 128, 128)
