
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, query, key, value, scale_factor, dropout_p=0.0):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Input
query = torch.randn(1, 4, 512)
key = torch.randn(1, 5, 512)
value = torch.randn(1, 5, 512)
scale_factor = torch.randn(1, 4, 1)
dropout_p = 0.1
