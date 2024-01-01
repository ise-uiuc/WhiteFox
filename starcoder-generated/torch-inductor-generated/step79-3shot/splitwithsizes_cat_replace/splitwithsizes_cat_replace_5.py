
class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(11025, 11025, bias=False)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1024, 128, 256, 256, 512, 512, 512, 512, 1024, 1024, 1024, 1024, 4096, 4096, 4096, 4096], dim=-1)
        concatenated_tensor = torch.cat(split_tensors, dim=-1)
        return (concatenated_tensor, torch.split(v1, [1024, 128, 256, 256, 512, 512, 512, 512, 1024, 1024, 1024, 1024, 4096, 4096, 4096, 4096], dim=-1))
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        module = [Block()]
        self.features = torch.nn.Sequential(*module)
        self.extra = torch.nn.ReLU()
    def forward(self, v1):
        split_tensors = torch.split(v1, [1025, 129, 257, 257, 513, 513, 513, 513, 1025, 1025, 1025, 1025, 4097, 4097, 4097, 4097], dim=3)
        concatenated_tensor = torch.cat(split_tensors, dim=3)
        return (concatenated_tensor, torch.split(v1, [1025, 129, 257, 257, 513, 513, 513, 513, 1025, 1025, 1025, 1025, 4097, 4097, 4097, 4097], dim=3))
# Inputs to the model
x1 = torch.randn(1, 3, 14, 14)
