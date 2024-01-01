
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, query, key, value, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
        if query.size(-1)!= key.size(-1):
            raise ValueError('Query and key must have the same "time dimension".'
                             + 'Found query with size'+ str(query.size())
                             + 'and key with size'+ str(key.size()))
        if key.size(-2)!= value.size(-2):
            raise ValueError('Key and value must have the same "feature dimension".'
                             + 'Found key with size'+ str(key.size())
                             + 'and value with size'+ str(value.size()))
        self.scale_factor = torch.sqrt(torch.FloatTensor([query.size(-1)]))

    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk * self.scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
query = torch.randn(1, 8, 16)
key   = torch.randn(1, 8, 8)
value = torch.randn(1, 8, 16)
dropout_p = 0.5
m = Model(query, key, value, dropout_p)

# Inputs to the model
