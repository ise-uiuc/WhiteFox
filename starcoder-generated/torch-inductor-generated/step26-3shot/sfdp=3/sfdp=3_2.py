
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # scale_factor is a trainable parameter
        self.scale_factor = torch.nn.Parameter(torch.ones(1))
        # dropout_p is a trainable parameter
        self.dropout_p = torch.nn.Parameter(torch.tensor(0.))
 
    def forward(self, query):
        # Compute the dot product between the query and key tensors
        # This method does not contain the value tensor since it is passed as an argument
        k = query
        qk = torch.matmul(query, k.transpose(-2, -1))
        # Scale the dot product by the trainable parameter scale_factor
        scaled_qk = qk * self.scale_factor
        # Apply softmax to the scaled dot product
        softmax_qk = scaled_qk.softmax(dim=-1)
        # Apply dropout to the softmax output
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        # Compute the dot product of the dropout output and the value tensor
        value = torch.randn(softmax_qk.size())
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

x1 = torch.randn(1, 196, 13)
