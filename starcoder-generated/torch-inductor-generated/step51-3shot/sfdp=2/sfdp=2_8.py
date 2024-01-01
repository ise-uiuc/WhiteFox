
class Model(torch.nn.Module):
    def __init__(self, dim_model, num_heads, dropout_p):
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dropout_p = dropout_p

    def forward(self, query, key, value, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output
        
# Initializing the model
model = Model(dim_model=16, num_heads=4, dropout_p=0.75)

# Inputs to the model
query = torch.randn(1, 16, 512, 8)
key = torch.randn(1, 16, 896, 8)
value = torch.randn(1, 16, 896, 8)
