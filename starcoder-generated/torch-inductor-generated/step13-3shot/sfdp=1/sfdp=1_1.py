
class Model(torch.nn.Module):
    def __init__(self, dim_k):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, query, key, value, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(dim_k=2)

# Inputs to the model
query = torch.randn(1, 2, 10)
key = torch.randn(1, 5, 2)
value = torch.randn(1, 5, 10)
inv_scale_factor = torch.randn(1, 2, 10)
