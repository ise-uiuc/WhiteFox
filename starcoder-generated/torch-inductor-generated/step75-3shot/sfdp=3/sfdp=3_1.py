
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query1, key1, value1, scale_factor1, dropout_p1):
        qk = torch.matmul(query1, key1.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor1)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p1)
        output = dropout_qk.matmul(value1)
        return output
 
model = Model()
 
# Inputs to the model
query1 = torch.ones(1, 8, 32, 32)
key1 = torch.ones(1, 8, 32, 32)
value1 = torch.ones(1, 8, 32, 32)
scale_factor1 = torch.ones(8, 8) * 0.5
dropout_p1 = 0
