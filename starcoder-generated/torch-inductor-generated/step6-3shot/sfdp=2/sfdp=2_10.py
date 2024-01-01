
class Model(torch.nn.Module):
    def __init__(self):
        pass  # leave it empty
    def forward(self, query_t, key_t, value_t):
        v1 = torch.matmul(query_t, key_t.transpose(-2, -1))
        v2 = v1.div(float(1.0 / self.scale_factor))
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=self.dropout_p, training=self.training)
        output = torch.matmul(v4, value_t)
        return output

# Initializing the model
m = Model()
m.scale_factor = 10
m.dropout_p = 0.5

# Inputs to the model
query_t = torch.randn(1, 100, 128)
key_t = torch.randn(1, 128, 100)
value_t = torch.randn(1, 128, 100)
