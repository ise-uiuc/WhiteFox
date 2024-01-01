
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dot = torch.nn.QFunctional()
 
    def forward(self, q, k, v, training):
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = 0.125

        softmax_qk = self.dot.softmax(qk / inv_scale_factor, dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, training=training)
        output = torch.matmul(dropout_qk, v)
        return output

# Initializing the model
model = Model()

# Inputs to the model
query = torch.randn(1, 16, 128, 32)
key = torch.randn(1, 16, 256, 64)
value = torch.randn(1, 16, 256, 64)
training = True

