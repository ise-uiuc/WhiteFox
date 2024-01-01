
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output_size = 256

    def forward(self, query, key, value, query_scale_factor=1.0, key_scale_factor=1.0, value_scale_factor=1.0):
        qk = torch.matmul(query, key.transpose(-2, -1)) * query_scale_factor * key_scale_factor
        scaled_qk = torch.div(qk, value_scale_factor)
        softmax_qk = torch.nn.functional.softmax(scaled_qk, dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.4)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(8, 256, 30)
key = torch.randn(8, 30, 256)
value = torch.randn(8, 256, 256)
