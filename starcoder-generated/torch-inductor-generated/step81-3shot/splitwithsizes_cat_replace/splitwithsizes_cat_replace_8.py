
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        block = [torch.nn.Conv2d(3, 32, 3, 1, 1, bias=False)]
        self.features = torch.nn.Sequential(*block)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        block = [torch.nn.Conv2d(32, 8, 3, 1, 1, bias=False)]
        self.features = torch.nn.Sequential(*block)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
class Model3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        block = [torch.nn.Conv2d(8, 8, 3, 1, 1, bias=False), torch.nn.ReLU()]
        self.features = torch.nn.Sequential(*block)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
class Model4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        block = [torch.nn.Conv2d(16, 64, 3, 1, 1, bias=False)]
        self.features = torch.nn.Sequential(*block)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1, 1], dim=1))
class Model5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        block = [torch.nn.Conv2d(8, 32, (3, 5), 1, (1, 2), bias=False)]
        self.features = torch.nn.Sequential(*block)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
class Model6(torch.nn.Module):
    def __init__(self):
        super().__init__()
        block = [torch.nn.Conv2d(8, 32, (3), 1, (1), dilation=2, bias=False)]
        self.features = torch.nn.Sequential(*block)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
    return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
class Model7(torch.nn.Module):
    def __init__(self):
        super().__init__()
        block = [torch.nn.Conv2d(32, 64, (2, 4), (1, 2), (0, 1), bias=False)]
        self.features = torch.nn.Sequential(*block)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1, 1, 1, 1, 1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1, 1, 1, 1, 1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
