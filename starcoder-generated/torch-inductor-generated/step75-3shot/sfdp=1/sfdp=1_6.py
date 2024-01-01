
class Model(torch.nn.Module):
    def __init__(self, dim, num_heads, dropout_p=0.5):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(dim, num_heads, dropout_p, batch_first=True)
 
    def forward(self, query, key, value, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output
    
# Initialize the model
m  = Model(dim=256, num_heads=8)

# Inputs to the model
x1 = torch.randn(16, 4, 256)
x2 = torch.randn(16, 6, 256)
x3 = torch.randn(16, 6, 256)
inv_scale_factor = 1/8
