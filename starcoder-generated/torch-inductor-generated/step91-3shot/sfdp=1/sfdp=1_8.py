
class Model(torch.nn.Module):
    def __init__(self, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
 
    def forward(self, query, key, value, inv_scale_factor=1.0, dropout_p=0.5):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Inputs to the model
query = torch.randn(1, 2, 3, 4)
key = torch.randn(1, 2, 4, 3)
value = torch.randn(1, 2, 4, 5)
inv_scale_factor = 1.0
dropout_p = 0.5

