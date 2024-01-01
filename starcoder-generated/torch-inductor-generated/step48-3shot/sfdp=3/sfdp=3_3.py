
class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.25)
 
    def forward(self, query, key, value, scale_factor=None):
        global attention_mask
        qk = torch.matmul(query, key.transpose(-2, -1))
        if scale_factor is not None:
            scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        attention_mask = softmax_qk
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Attention()

# Inputs to the model
query = torch.randn(1, 8, 64, 64)
key = torch.randn(1, 8, 64, 64)
value = torch.randn(1, 8, 64, 64)
scale_factor = torch.randn(1, 8, 1, 1)
