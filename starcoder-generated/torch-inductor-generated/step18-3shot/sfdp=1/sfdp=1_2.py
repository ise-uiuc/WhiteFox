
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transpose = torch.nn.Conv2d(1, 1, 1, stride=1, padding=1)
        self.dropout = torch.nn.Conv2d(1, 1, 1, stride=1, padding=1)
 
    # query : input1
    # key : input2
    # value : input3
    # inv_scale_factor : input4
    # dropout_p : input5
    def forward(self, input1, input2, input3, input4, input5):
        qk = torch.matmul(input1, input2)
        scaled_qk = qk.div(input4) # Scale the dot product by the inverse scale factor
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        output = self.dropout(softmax_qk.matmul(input3)) # Compute the dot product of the dropout output and the value tensor
        return output
        
# Initializing the model
m = Model()

# Inputs to the model
input1 = torch.randn(1, 1, 64, 64)
input2 = torch.randn(1, 1, 64, 64)
input3 = torch.randn(1, 1, 64, 64)
input4 = 1
input5 = 0.5
