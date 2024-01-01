
class Model(torch.nn.Module):
    def __init__(self, dropout_p=0.7, scale_factor=1.0):
        super().__init__()
        self.dropout_p = dropout_p
        self.scale_factor = scale_factor

    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output
 
# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 5, 10, 16)
key = torch.randn(1, 5, 12, 16)
value = torch.randn(1, 5, 12, 16)
