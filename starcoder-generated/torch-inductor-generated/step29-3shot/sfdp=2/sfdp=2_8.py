
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, input,
                weight_0, weight_1,
                bias_0, bias_1,
                dropout_p):
        v0 = torch.matmul(input, weight_0)
        v0 = v0 + bias_0
        v0 = torch.nn.functional.silu(v0)
        v1 = torch.matmul(v0, weight_1)
        v1 = v1 + bias_1
        v1 = torch.nn.functional.dropout(v1, p=dropout_p)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(4, 64, 1024)
weight_0 = torch.randn(4, 64, 3072)
weight_1 = torch.randn(4, 3072, 1536)
bias_0 = torch.randn(4, 64, 3072)
bias_1 = torch.randn(4, 3072, 1536)
dropout_p = 0.25
__output = m(input,
            weight_0, weight_1,
            bias_0, bias_1,
            dropout_p)


# Model 2
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, input_tensor, mlp_input_tensor):
        v0 = input_tensor * 0.5
        v1 = 2 * mlp_input_tensor
        v2 = v0 * v1
        v3 = 9 * mlp_input_tensor
        v4 = 3 * v3
        v5 = v2 + v4
        v6 = 7 * mlp_input_tensor
        v7 = torch.erf(v6)
        v8 = v7 + 1
        v9 = v5 * v8
        return v9
 
# Initializing the model
m = Model()
 
# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)
mlp_input_tensor = torch.randn(1, 3, 64, 64)
