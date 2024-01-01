
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax(dim=-1)
 
    def forward(self, query, key, value):
        qk = torch.matmul(query_tensor, key_tensor.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value_tensor)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 3, 8, 8)
key = torch.randn(1, 3, 16, 16)
value = torch.randn(1, 3, 16, 16)
