
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, scale_factor, dropout_p):
        output = torch.matmul(query, key.transpose(-2, -1))
        output = output.div(scale_factor)
        output = torch.nn.functional.softmax(output)
        output = torch.nn.functional.dropout(output, p=dropout_p)
        output = torch.matmul(output, value)
        return output

# Initializing the model
q, k, v = torch.randn(1, 5, 16), torch.randn(1, 5, 24), torch.randn(1, 5, 24)
s = torch.randn(16, 24)
dp = 0.1
model = Model()
