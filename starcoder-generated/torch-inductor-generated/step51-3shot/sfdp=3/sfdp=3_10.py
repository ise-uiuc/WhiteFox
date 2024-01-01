
class Model(torch.nn.Module):
    def __init__(self, dropout_p, scale_factor):
        super().__init__()
        self.dropout_p = dropout_p
        self.scale_factor = scale_factor
 
    def forward(self, query, key, value):
        scaled_qk = torch.matmul(query, key.transpose(-2, -1)).mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(0.5, 0.5)

# Input to the model
m(query, key, value)

# Inputs to the model
query = torch.randn(1, 1, 5, 3)
key = torch.randn(1, 1, 3, 3)
value = torch.randn(1, 1, 3, 5)
