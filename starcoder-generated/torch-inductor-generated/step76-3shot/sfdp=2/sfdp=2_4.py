
class Model(torch.nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.query = torch.nn.Linear(input_shape[1], input_shape[1])
        self.key = torch.nn.Linear(input_shape[1], input_shape[1])
        self.value = torch.nn.Linear(input_shape[1], input_shape[1])
        self.dropout = torch.nn.Dropout(0.1)
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

input_shape = [1, 3, 64]
model = Model(input_shape)

# Inputs to the model
query = torch.randn(1, 3, 64)
key = torch.randn(1, 3, 64)
value = torch.randn(1, 3, 64)
