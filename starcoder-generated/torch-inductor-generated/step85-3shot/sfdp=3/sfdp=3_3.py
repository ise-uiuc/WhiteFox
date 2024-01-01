
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def scaled_product(self, query, key, value, scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        return scaled_qk

    def forward(self, query, key, value, scale_factor, dropout_p):
        scaled_qk = self.scaled_softmax(query, key, value, scale_factor)
        dropout_qk = torch.nn.functional.dropout(scaled_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(2, 128, 32, 256)
key = torch.randn(2, 128, 32, 256)
value = torch.randn(2, 128, 32, 256)
