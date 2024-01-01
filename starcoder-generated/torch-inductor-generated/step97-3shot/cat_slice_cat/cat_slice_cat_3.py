
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, size, pad_value1, pad_value2, pad_value3, mode, data_format):
        v1 = torch.cat([x1, x2, x3], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:size]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 3, 224, 224, requires_grad=False)
x2 = torch.randn(2, 3, 224, 224, requires_grad=False)
x3 = torch.randn(2, 3, 224, 224, requires_grad=False)
size = torch.tensor(3, dtype=torch.int64)
pad_value1 = torch.tensor(0, dtype=torch.float32)
pad_value2 = torch.randn(6144, dtype=torch.float32, requires_grad=False)
pad_value3 = torch.randn([(size[0] - x1.shape[1])], dtype=torch.float32, requires_grad=False)
torch.manual_seed(0)
mode = torch.tensor(0, dtype=torch.int64)
data_format = torch.tensor(0, dtype=torch.int64)

