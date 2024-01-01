
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = math.sqrt(8 * 3)
 
    def forward(self, query, key1, value1):
        qk = torch.matmul(query, key1.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value1)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(16, 3, 64)
key1 = torch.randn(16, 8, 64)
value1 = torch.randn(16, 8, 64)
