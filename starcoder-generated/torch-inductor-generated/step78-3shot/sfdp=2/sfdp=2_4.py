
class Model(torch.nn.Module):
    def __init__(self, query, key, value, scale_factor, dropout_p):
        super().__init__()
        self.scale_factor = scale_factor
        self.dropout_p = dropout_p
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, query, key, value):
        k = torch.matmul(query, key.transpose(-2, -1))
        v = torch.matmul(query, value.transpose(-2, -1))
        scaled_k = k.div(self.scale_factor)
        softmax_k = scaled_k.softmax(dim=-1)
        dropout_softmax_k = self.dropout(softmax_k)
        output = dropout_softmax_k.matmul(v)
        return output
 
# Initializing the model
scale_factor = 2 ** 0.5
dropout_p = 0.8
m = Model(query, key, value, scale_factor, dropout_p)

# Inputs to the model
x1 = torch.randn(1, 4, 20)
x2 = torch.randn(1, 2, 20)
x3 = torch.randn(1, 20, 1)
