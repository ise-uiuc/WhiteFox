
class Model(torch.nn.Module):
    def __init__(self, inv_scale_factor, dropout_p):
        super().__init__()
        self.inv_scale_factor = inv_scale_factor
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value):
        _matmul1 = query.matmul(key.transpose(-1, -2))
        _div1 = _matmul1 / self.inv_scale_factor
        _softmax1 = torch.softmax(_div1, -1)
        _dropout1 = torch.nn.functional.dropout(_softmax1, self.dropout_p, True,)
        _matmul2 = _dropout1.matmul(value)
        