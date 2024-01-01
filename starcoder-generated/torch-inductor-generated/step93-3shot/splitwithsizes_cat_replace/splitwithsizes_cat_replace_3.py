
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = torch.nn.ModuleList([torch.nn.Conv2d(3, out_channels=32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False), torch.nn.Conv2d(32, out_channels=16, kernel_size=(1, 1), stride=(1, 1), bias=False)])
        self.features2 = torch.nn.ModuleList([torch.nn.Conv2d(16, out_channels=32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False), torch.nn.Conv2d(32, out_channels=32, kernel_size=(3, 1), stride=(1, 1), bias=False)])
        self.features3 = torch.nn.ModuleList([torch.nn.Conv2d(32, out_channels=48, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False), torch.nn.Conv2d(48, out_channels=32, kernel_size=(5, 1), stride=(1, 1), bias=False)])
        self.features4 = torch.nn.ModuleList([torch.nn.Conv2d(32, out_channels=72, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False), torch.nn.Conv2d(72, out_channels=32, kernel_size=(7, 1), stride=(1, 1), bias=False)])
    def forward(self, v1):
        self.split_tensors = torch.split(v1, [1, 1], dim=-1)
        self.split_tensors[1] = torch.squeeze(self.split_tensors[1], dim=-1)
        self.concatenated_tensor = torch.concat([self.split_tensors[0], self.split_tensors[1]], dim=-1)
        self.intermediate_layer_output = torch.unsqueeze(self.concatenated_tensor, 1)
        for j in range(len(self.features) - 1):
            self.features2[j](self.intermediate_layer_output)
            j += 1
        return (self.concatenated_tensor, torch.split(v1, [1, 1], dim=-1))
# Inputs to the model
x1 = torch.randn(1, 64, 3, 3)
