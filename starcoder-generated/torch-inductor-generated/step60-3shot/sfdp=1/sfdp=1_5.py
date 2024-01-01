
self.softmax_qk = torch.nn.Softmax(dim=-1)
self.dropout = torch.nn.Dropout(dropout_p)
...
def forward(self, query, key, value, inv_scale_factor, dropout_p):
    qk = torch.matmul(query, key.transpose(-2, -1))
    scaled_qk = qk.div(inv_scale_factor)
    softmax_qk = self.softmax_qk(scaled_qk)
    dropout_qk = self.dropout(softmax_qk)
    output = dropout_qk.matmul(value)
    return output