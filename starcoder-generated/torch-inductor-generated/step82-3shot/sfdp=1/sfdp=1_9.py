
class Model(torch.nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
 
    @staticmethod
    def shape(x):
        return torch.cat([torch.tensor(x.shape[0:2] + [1]),
                          torch.tensor(x.shape[2:4] + [1]),
                          torch.tensor(x.shape[4:]),torch.tensor(x.shape[0:2] + [1]),
                          torch.tensor(x.shape[2:4] + [1]),
                          torch.tensor(x.shape[4:])], dim=0)
 
    def forward(self, query, key, value, dropout_p, scale_factor=None):
        if scale_factor is None:
            scale_factor = 1 / math.sqrt(query.dim())
        qk = torch.matmul(query, key.transpose(-2, -1))
        qk = torch.reshape(qk, self.shape(qk))
        scaled_qk = qk.div(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        dropout_qk = torch.reshape(dropout_qk, self.shape(dropout_qk))
        output = torch.matmul(dropout_qk, value)
        return output

# Initializing the model
num_heads = 4
m = Model(num_heads)

# Inputs to the model
query = torch.randn(1, num_heads, 192, 128)
key = torch.randn(1, num_heads, 256, 128)
value = torch.randn(1, num_heads, 256, 128)
dropout_p = 0.5
