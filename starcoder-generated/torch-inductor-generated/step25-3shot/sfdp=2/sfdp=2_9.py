
class Model(torch.nn.Module):
    def __init__(self, dropout_p, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(0.1, 4)

# Inputs to the model
query = torch.randn(1, 4, 256)
key = torch.randn(1, 4, 256)
value = torch.randn(1, 4, 256)
