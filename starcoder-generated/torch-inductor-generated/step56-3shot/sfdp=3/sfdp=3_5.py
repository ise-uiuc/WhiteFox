
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, scale_factor, value, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output
 
m = Model()
query = torch.randn(1, 10, 5)
key = torch.randn(1, 20, 5)
scale_factor = torch.randn(1, 10, 1)
value = torch.randn(1, 20, 5)
dropout_p = 0.05
