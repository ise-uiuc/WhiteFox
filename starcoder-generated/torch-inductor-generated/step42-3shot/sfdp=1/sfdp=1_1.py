
class Model(torch.nn.Module):
    def __init__(self, depth, dropout_p):
        super().__init__()
        self.depth = depth
        self.dropout_p = dropout_p 

    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = 1 / math.sqrt(math.sqrt(self.depth))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
depth = __depth_value__
dropout_p = __dropout_p_value__
m = Model(depth, dropout_p)

# Inputs to the model
query = torch.randn(1, 2, __depth_value__, __depth_value__)
key = torch.randn(1, 2, __depth_value__, __depth_value__)
value = torch.randn(1, 2, __depth_value__, __depth_value__)
