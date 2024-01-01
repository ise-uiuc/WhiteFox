
class Model(torch.nn.Module):
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scale_factor = torch.tensor(0.1)
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()
# Input tensors to the model
query = torch.randn(5, 3, 8)
key = torch.randn(4, 3, 8)
value = torch.randn(5, 4, 8)
