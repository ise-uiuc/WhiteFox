
class Model(torch.nn.Module):
    def forward(self, query, key, value,
                inv_scale_factor=1.0 / math.sqrt(0.5),
                dropout_p=0):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        return dropout_qk.matmul(value)

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 8, 128, 256)
key = torch.randn(1, 8, 128, 256)
value = torch.randn(1, 8, 128, 256)
