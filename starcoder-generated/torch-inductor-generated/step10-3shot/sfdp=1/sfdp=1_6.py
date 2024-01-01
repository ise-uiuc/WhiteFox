
class Model(torch.nn.Module):
    def __init__(self, query, key, value, dropout_p, inv_scale_factor):
        super().__init__()
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1) # [B, N, 64, 512]
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p) # [B, N, 64, 512]
        output = dropout_qk.matmul(value) # [B, N, 64, 512]
        return output

# Initializing the model
m = Model(query, key, value, dropout_p, inv_scale_factor)

# Inputs to the model
query = torch.randn(1, 64, 512)
key = torch.randn(1, 512, 64)
value = torch.randn(1, 512, 64)
