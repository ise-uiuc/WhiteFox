
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, 1, 1), torch.nn.Conv2d(32, 32, 4, 1, 5))
        self.split = torch.nn.Sequential()
        self.concat = torch.nn.Sequential()
    def forward(self, v1):
        split_tensors = []
        for i in range(3):
            split_tensors.append(torch.split(v1, [1, 1, 1], dim=1))
        for i in range(len(split_tensors)):
            for j in range(3):
                if split_tensors[i][j].shape[1] > 1:
                    split_tensors[i][j] = torch.split(split_tensors[i][j], [1, 1, 1], dim=1)
        concatenated_tensor = []
        for i in range(3):
            concatenated_tensor.append(torch.cat([(split_tensors[i][j][0] for j in range(3))], dim=1))
        for i in range(len(split_tensors)):
            for j in range(3):
                if split_tensors[i][j].shape[1] > 1:
                    concatenated_tensor[i] = torch.cat([concatenated_tensor[i], torch.split(concatenated_tensor[i], [1, 1, 1], dim=1)], dim=1)
        return (torch.cat(concatenated_tensor, dim=1), split_tensors)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
