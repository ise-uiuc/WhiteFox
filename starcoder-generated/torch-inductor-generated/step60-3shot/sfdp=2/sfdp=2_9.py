
class Model1(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        shape_qk = (in_features, out_features) # Shape of query and key
        shape_v = out_features # Shape of value
        self.shape_qk = torch.Size(shape_qk)
        self.shape_v = torch.Size(shape_v)
        shape_logits = () # Shape of the logits
        self.shape_logits = torch.Size(shape_logits)
        inv_shape_logits = ()
        self.inv_scale_factor = 1
        self.dropout_p = 0
        shape_softmax_logits = (shape_qk[-2], shape_qk[-1]) # Shape of the softmax logits
        self.shape_softmax_logits = torch.Size(shape_softmax_logits)
        shape_dropout_qk = (shape_qk[-2], shape_qk[-1]) # Shape of the dropout qk
        self.shape_dropout_qk = torch.Size(shape_dropout_qk)
        
    def forward(query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1)) # Compute the dot product of the query and the key
        scaled_qk = qk.div(self.inv_scale_factor) # Scale the dot product by the inverse scale factor
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p) # Apply dropout to the softmax output
        output = dropout_qk.matmul(value) # Compute the dot product of the dropout output and the value
        return output
  
model = Model1(5, 2)

# Inputs to the model
query = torch.randn(1, 4, 5)
key = torch.randn(1, 8, 5)
value = torch.randn(1, 8, 2)
