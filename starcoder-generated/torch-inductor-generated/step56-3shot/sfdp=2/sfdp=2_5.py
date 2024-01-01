
class Model(torch.nn.Module):
    def __init__(self, *, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value, scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(dropout_p=0.3)

# Inputs to the model
query = torch.randn(5, 16, 100)
key = torch.randn(5, 256, 16)
value = torch.randn(5, 256, 100)
scale_factor = torch.empty(5, 256).fill_(1).uniform_(0, 1)
