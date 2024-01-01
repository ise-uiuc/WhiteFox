
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x1, x2, x3):
        x4 = torch.matmul(x1, x2.T)
        x5 = x4 * self.scale_factor
        x6 = x5.softmax(dim=-1)
        x7 = x6 * self.dropout_p
        x8 = torch.matmul(x7, x3)
        return x8

# Initializing the model
m = Model()

# Inputs to the model
input_tensor_x1 = torch.randn(1, 3, 64, 64)
input_tensor_x2 = torch.randn(1, 3, 64, 64)
input_tensor_x3 = torch.randn(1, 3, 64, 64)

