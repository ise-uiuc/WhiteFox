
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        v1 = torch.nn.functional.linear(x1, x2)
        v2 = v1 - x2
        return v1

# Initializing the model

# __nnfw_op_t_0 [FLOAT32 5 5 15 15] 
# __nnfw_op_t_1 [FLOAT32 10 1 28 28]

# Inputs to the model
x1 = torch.randn(1, 5, 5, 15, 15)
x2 = torch.randn(1, 10, 1, 28, 28)

