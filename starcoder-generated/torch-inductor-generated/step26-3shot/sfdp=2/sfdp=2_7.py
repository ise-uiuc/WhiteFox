
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(10.0)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.75)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
input_tensor = torch.randn(1, 2, 2)
query = torch.randn(1, 1, 2)
key = torch.randn(1, 2, 2)
value = torch.randn(1, 2, 2)
