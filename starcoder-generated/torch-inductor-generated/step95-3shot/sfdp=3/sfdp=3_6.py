
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout_p)
 
    def forward(self, query, key, value, scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = torch.nn.functional.softmax(scaled_qk, dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
model = Model()

# Inputs to the model
query = torch.randn(4, 5, 15)
key = torch.randn(4, 5, 15)
value = torch.randn(4, 6, 15)
scale_factor = 1 / np.sqrt(value_head.size(1))
 
