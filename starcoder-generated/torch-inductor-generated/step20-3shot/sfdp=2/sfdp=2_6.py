
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.5
        self.scale_factor = math.sqrt(0.5)
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.t()).div(self.scale_factor)
        softmax_qk = qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 4, 100)
key = torch.rand(1, 4, 200)
value = torch.randn(1, 4, 200)
