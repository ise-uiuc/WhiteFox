
batch_size = 1
channel = 1
head_num = 12
attention_head_size = 64
sequence_length = 1024

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # Initialize the parameter
        self.scale_factor = math.sqrt(attention_head_size)
    
    def forward(self, query, key, value, attention_mask):
        # Apply qk
        qk = torch.matmul(query, key.transpose(-2, -1)) # qk.shape: [batch_size, head_num, sequence_length, sequence_length]
        
        # Scale the qk
        inv_scale_factor = 1 / self.scale_factor
        scaled_qk = qk.div(inv_scale_factor)
        
        # Apply softmax
        softmax_qk = scaled_qk.softmax(dim=-1) # softmax_qk.shape: [batch_size, head_num, sequence_length, sequence_length]
        
        # Apply dropout
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1) # dropout_qk.shape: [batch_size, head_num, sequence_length, sequence_length]
        
        # Apply mv
        output = dropout_qk.matmul(value) # output.shape: [batch_size, head_num*attention_head_size, sequence_length]
        return output

# Initialize the input tensor
query = torch.randn(batch_size, head_num, attention_head_size, sequence_length)
key = torch.randn(batch_size, head_num, attention_head_size, sequence_length)
value = torch.randn(batch_size, head_num, attention_head_size, sequence_length)
attention_mask = torch.ones([batch_size, head_num, sequence_length, sequence_length])
attention_mask[0][0][:42] = 0
attention_mask[0][0][42:84] = 0
attention_mask[0][0][84:] = 0

# Initialize and run the model
m = Model()

result = m(query, key, value, attention_mask)

# Print result
print(result)

