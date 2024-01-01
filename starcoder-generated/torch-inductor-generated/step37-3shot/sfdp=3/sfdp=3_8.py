
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = math.sqrt(dim_head)
 
    def forward(self, query, key, value, dropout_p):
        qk_mul = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk_mul * self.scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(16, dim_head, 8, 64)
key = torch.randn(16, dim_head, 8, 64)
value = torch.randn(16, dim_head, 8, 64)
dropout_p = 0.3
