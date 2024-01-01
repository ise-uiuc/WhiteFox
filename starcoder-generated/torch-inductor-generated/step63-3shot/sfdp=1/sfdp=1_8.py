
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.randn(d_keys, d_model)
        self.key = torch.randn(d_keys, d_model)
        self.value = torch.randn(d_values, d_model)
 
    def forward(self, x1):
        qk = torch.matmul(self.query, self.key.transpose(-2, -1))
        scale_factor = d_keys ** -0.5
        scaled_qk = qk.div(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, 0.1)
        output = dropout_qk.matmul(self.value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, d_model)
