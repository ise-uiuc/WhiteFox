
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # model parameters
        self.split_output_size = 3
        # end of model parameters

        self.features = nn.Sequential(
            convbn_3d(3, 1, kernel_size=3, stride=1, padding=1),
            convbn_3d(1, 1, kernel_size=3, stride=1, padding=1),
            convbn_3d(1, 1, kernel_size=3, stride=1, padding=1),
        )

        self.conv1 = nn.Conv2d(1, 1, 1, 1, 0, bias=False)
        self.conv2 = nn.Conv2d(1, 1, 3, 1, 1, bias=False)

    def forward(self, x):
        features = self.features(x)

        split_list = torch.split(features[0], [1, 1, 1], dim=0)
        split_list_sorted_1 = torch.stack(sorted(split_list, key=lambda split_list: split_list.max(dim=0).indices.item()))
        split_list_sorted_2 = torch.stack(list(reversed(split_list)))
        reordered_tensors = torch.stack([split_list_sorted_1[self.split_output_size-1-i] for i in range(self.split_output_size)])

        y1 = self.conv1(features[0])
        y1_split_tensors = torch.split(y1, [1, 1, 1], dim=0)
        y1_concatenated_tensor = torch.cat(y1_split_tensors, dim=0)
        y2 = self.conv2(features[0])
        y2_split_tensors = torch.split(y2, [1, 1, 1], dim=0)
        y2_concatenated_tensor = torch.cat(y2_split_tensors, dim=0)

        concatenated_tensor = torch.cat((reordered_tensors, y1_concatenated_tensor, y2_concatenated_tensor), dim=0)

        return concatenated_tensor, reordered_tensors
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
