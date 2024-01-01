
class Model(torch.nn.Module):
    def __init__(self, attention_mask_shape):
        super().__init__()
 
        # Create the scale factor from the attention mask shape
        num_heads = attention_mask_shape[1]
        mask_sequence_length = attention_mask_shape[2]
        scale_factor = torch.sqrt(torch.Tensor([num_heads * mask_sequence_length]))
 
        # Create parameters that can be used by all heads
        self.scale_factor = torch.nn.Parameter(scale_factor, requires_grad=False)
 
        # Create parameters that will be used by one head
        self.dropout_p = torch.nn.Parameter(torch.tensor(0.1), requires_grad=True)
 
    # Forward pass for one head
    def forward(self, query, key, value, attention_mask):
 
        # Compute the dot product of the query and key tensors
        qk = torch.matmul(query, key.transpose(-2, -1))
 
        # Scale the dot product by the inverse scale factor
        scaled_qk = qk.div(self.scale_factor)
 
        # Apply softmax to the scaled dot product
        softmax_qk = scaled_qk.softmax(dim=-1)
 
        # Apply dropout to the softmax output
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
 
        # Compute the dot product of the dropout output and the value tensor
        output = dropout_qk.matmul(value)
 
        # Apply the attention mask
        return output * attention_mask.unsqueeze(1)

# Initializing the model
__m__ = Model((3, 8, 35))
 
# Inputs to the model
query = torch.randn(2, 8, 5, 64)
key = torch.randn(2, 8, 35, 64)
value = torch.randn(2, 8, 35, 64)
attention_mask = torch.randn(2, 8, 35)
