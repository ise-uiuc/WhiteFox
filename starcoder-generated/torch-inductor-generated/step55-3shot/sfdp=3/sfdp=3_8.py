
class Model(torch.nn.Module):
    def __init__(self, input_size=32, output_size=32):
        super().__init__()
        self.scale_factor = torch.full([input_size], pow(input_size, 0.5))
 
    def forward(self, query, key, value, dropout_p=0.1):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 1, 32)
key = torch.randn(1, 1, 32)
value = torch.randn(1, 1, 32)
