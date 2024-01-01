
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1
        self.dropout_p = 1
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1)) * self.scale_factor
        softmax_qk = qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 3, 300)
key = torch.randn(1, 3, 200)
value = torch.randn(1, 3, 200)
