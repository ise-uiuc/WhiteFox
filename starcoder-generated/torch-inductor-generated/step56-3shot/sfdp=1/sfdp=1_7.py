
class Model(torch.nn.Module):
    def __init__(self, query, key, value, inv_scale_factor, dropout_p):
        super().__init__()
        self.m1 = torch.nn.quantized.FloatFunctional()
        self.qk = torch.nn.quantized.FloatFunctional()
        self.dropout = torch.nn.quantized.FloatFunctional()
        self.m2 = torch.nn.quantized.FloatFunctional()

    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        qk = self.m1.mul(query, key.transpose(-2, -1))
        scaled_qk = self.qk.div(qk, inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout.dropout(softmax_qk, p=dropout_p)
        output = self.m2.mul(dropout_qk, value)
        return output

# Initializing the model
query = torch.randn(128, 128)
key = torch.randn(128, 128)
value = torch.randn(128, 128)
inv_scale_factor = torch.randn(1)
dropout_p = torch.nn.quantized.FloatFunctional()
output = dropout_p.dropout(inv_scale_factor, p=0.5)
model = Model(query, key, value, output)

# Inputs to the model
query = torch.randn(128, 128)
key = torch.randn(128, 128)
value = torch.randn(128, 128)
inv_scale_factor = torch.randn(1)
dropout_p = torch.nn.quantized.FloatFunctional()
