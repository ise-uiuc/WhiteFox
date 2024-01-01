
class Model(torch.nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = torch.sqrt(torch.tensor(float(self.num_heads)).float())
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(8)

# Inputs to the model
query = torch.randn(1, 8, 64)
key = torch.randn(1, 8, 64)
value = torch.randn(1, 8, 64)
