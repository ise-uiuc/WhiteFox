
class Model(torch.nn.Module):
    def __init__(self, scale_factor, dropout_p):
        super().__init__()
        self.scale_factor = scale_factor
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(scale_factor=1 / input_tensor.size(0) ** 0.5, dropout_p=0.2)

# Inputs of the model
query = torch.randn(1, 3, 10)
key = torch.randn(1, 3, 10)
value = torch.randn(1, 3, 10)
