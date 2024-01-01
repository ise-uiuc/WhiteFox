
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scale_factor = torch.sqrt(qk.size(-1))
        qk = qk / scale_factor
        softmax_qk = qk.softmax(dim=-1)
 
        dropout_p = torch.empty(1).uniform_()
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 4, 64, 64)
key = torch.randn(1, 4, 512, 64)
value = torch.randn(1, 4, 512, 64)
