
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, dropout_p=0.20217948716018677, scale_factor=temperature):
        input_shape = query.shape
        key = key.transpose(-2, -1)
        qk = torch.matmul(query, key)
        inv_scale_factor = 1 / scale_factor
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(2, num_heads, 256, 64)
key = torch.randn(2, num_heads, 64, 256)
value = torch.randn(2, num_heads, 64, 256)
dropout_p = 0.20217948716018677
scale_factor = temperature # The temperature is a constant passed in as an argument by the user when using the model 

