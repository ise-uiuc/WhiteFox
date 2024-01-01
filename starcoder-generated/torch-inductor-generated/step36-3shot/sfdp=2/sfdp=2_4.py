
class Model(torch.nn.Module):
    def __init__(self, batch_size, sequence_length, embedding_size, head_num):
        super().__init__()
        self._batch_size = batch_size
        self._sequence_length = sequence_length
        self._embedding_size = embedding_size
 
        self.query = torch.nn.Parameter(torch.randn(batch_size, head_num, embedding_size))
        self.key = torch.nn.Parameter(torch.randn(batch_size, head_num, embedding_size))
        self.value = torch.nn.Parameter(torch.randn(batch_size, head_num, embedding_size))
 
    def forward(self, input, dropout_p):
        batch_size = self._batch_size
        sequence_length = self._sequence_length
        embedding_size = self._embedding_size
        head_num = self.query.size(1)
 
        q = self.query.permute(1, 0, 2)
        k = self.key.permute(1, 0, 2)
        v = self.value.permute(1, 0, 2)
 
        # Compute the dot product of the query and the key
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = math.sqrt(head_num * 1.0)
        scaled_qk = qk.div(inv_scale_factor)
 
        # Apply softmax to the scaled dot product
        softmax_qk = scaled_qk.softmax(dim=-1)
 
        # Apply dropout to the softmax output
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
 
        # Compute the dot product of the dropout output and the value
        output = dropout_qk.matmul(v)
 
        # Revert the permutation
        output = output.permute(1, 0, 2)
 
        # Reshape the output
        output = output.reshape((batch_size, sequence_length, embedding_size))
 
        return output, dropout_qk

# Initializing the model
m = Model(16, 32, 64, 8)

# Inputs to the model
input = torch.randn(16, 32, 64)
dropout_p = 0.5
output, dropout_qk = m(input, dropout_p)

