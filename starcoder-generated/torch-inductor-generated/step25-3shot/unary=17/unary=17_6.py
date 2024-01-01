
# Here i want only the tensor that is the result of conv_transpose, there are multiple ways to do this.
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 32, (8, 8), stride=(2, 2), padding=(5, 3), output_padding=(1, 0))
    def forward(self, x1):
       # First way 
        v1 = self.conv_transpose(x1)
        v2 = torch.relu(v1)
        return (v1.size()[0]) # Return size is used instead of v1 itself because by using size()[0] Tensor is not calculated.
       # Second way 
        v1 = self.conv_transpose(x1)
        list_v1 = v1.tolist() # This way convert the tensor into a list to be easy to manipulate
        list_v1_values = list_v1[0][0] # List of values of the matrix
        list_v1_rows = list_v1[0] # List of the rows
        list_v1_columns = list_v1[0][0] # List of the columns
        return len(list_v1[0][0]) # Return the number if elements of row 0 list 0
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
