
class Model(torch.nn.Module):
    def __init__(self, dropout_p=0.5, inv_scale_factor=1.0):
        super().__init__()
        self.dropout_p = dropout_p
        self.inv_scale_factor = inv_scale_factor
 
    def forward(self, query, key, value):
        matrix = torch.matmul(query, key.transpose(-2, -1))
        matrix_scale = matrix.div(self.inv_scale_factor)
        matrix_softmax = matrix_scale.softmax(dim=-1)
        output = torch.nn.functional.dropout(matrix_softmax, self.dropout_p).matmul(value)
        return output

# Initializing the model
m = Model(dropout_p=0.2, inv_scale_factor=100)

# Inputs to the model
t1 = torch.randn(1, 3, 32, 32)
t2 = torch.randn(1, 4, 32, 32)
t3 = torch.randn(1, 4, 32, 32)
