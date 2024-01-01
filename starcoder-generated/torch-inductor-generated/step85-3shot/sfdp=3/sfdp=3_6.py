
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, query, key, value, scale_factor, dropout_p, mask):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(4, 1, 128)
key = torch.randn(4, 129, 1)
value = torch.randn(4, 129, 128)
scale_factor = torch.tensor([1 / query.shape[-1] ** 0.25])
dropout_p = 0.2
mask = None
