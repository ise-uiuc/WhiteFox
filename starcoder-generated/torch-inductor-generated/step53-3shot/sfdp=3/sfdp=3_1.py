
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1.0 / math.sqrt(query_dim)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, query, key, value, mask):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        if mask is not None:
            output = output.masked_fill(mask.to(torch.bool), 0)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.uniform(-10., 10., (4, query_dim))
key = torch.uniform(-10., 10., (5, query_dim))
value = torch.uniform(-10., 10., (5, feature_dim))
mask = torch.tensor(
    [1, 1, 1, 1, 0], # Whether the value should be masked.
    device=query.device, dtype=torch.bool)
