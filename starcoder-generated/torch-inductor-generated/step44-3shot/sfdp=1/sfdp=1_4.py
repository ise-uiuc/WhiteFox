
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, scale_factor, dropout_p, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Creating random 3-D tensors used as input to the model. The dimensions of the tensors are (batch_size, head_number, length_q_or_k). The batch_size and head_number are picked from the set {1, 2, 4}, and the length_q_or_k is the length of the query and key tensors. If the shapes of query and key tensors are (batch_size, 8, 64), then the shapes of the query and key tensors used for initialization are (batch_size/head_number, 8*head_number, length_q_or_k), where the head_number is chosen from the set {1, 2, 4}.
head_number = 2

# The batch_size is 4 times the head_number
if head_number == 1:
    batch_size_for_init = 4
else:
    batch_size_for_init = head_number * 4

# The lengths of query and key are 64
length_q_or_k = 64
query=torch.rand((batch_size_for_init, head_number*8, length_q_or_k), dtype=torch.float32, requires_grad=True)
key=torch.rand((batch_size_for_init, head_number*8, length_q_or_k), dtype=torch.float32, requires_grad=True)

# The length of the value is 8, and the inverse of the scale factor is 128
scale_factor = 128.0
value=torch.rand((batch_size_for_init, head_number*8, 8), dtype=torch.float32, requires_grad=True)
inv_scale_factor = 1.0 / scale_factor

# The dropout probability is 0.9
dropout_p = 0.9

# Initialize the model
