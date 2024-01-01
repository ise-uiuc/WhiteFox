
class Model(torch.nn.Module):
    def forward(self, input_tensor, query_tensor, key_tensor, value_tensor):
        qk = torch.matmul(query_tensor, key_tensor.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value_tensor)
        return output

# Initializing the model
m = Model()

# Inputs to the model
input_tensor = torch.randn(1, 10, 100, 100)
query_tensor = torch.randn(1, 10, 400, 64)
key_tensor = torch.randn(1, 10, 64, 400)
value_tensor = torch.randn(1, 10, 400, 64)
