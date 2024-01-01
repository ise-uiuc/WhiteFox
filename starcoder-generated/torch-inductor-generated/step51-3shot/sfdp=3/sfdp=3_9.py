
class Model(torch.nn.Module):
    def __init__(self):
        self.dropout = torch.nn.Dropout()
 
    def forward(self, q, k, v, scale_factor, dropout_p):
        v = self.dropout(v)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Testing the model
torch.manual_seed(0)
m = Model()
