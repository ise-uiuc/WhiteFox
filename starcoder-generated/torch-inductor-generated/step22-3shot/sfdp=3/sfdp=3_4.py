
c = torch.randn(qk.size())
scaled_qk = query.matmul(key.transpose(-2, -1))
scaled_qk = scaled_qk * c
print(scaled_qk)
softmax_qk = scaled_qk.softmax(-1)
dropout_qk = nn.functional.dropout(softmax_qk, p=dropout_p)
print(dropout_qk)
output = dropout_qk.matmul(value)
print(output)

# Initializing the model
c = torch.randn(dropout_qk.size())
dropout_qk = dropout_qk * c
