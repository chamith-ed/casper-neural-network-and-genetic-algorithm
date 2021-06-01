import torch

# Used to convert dataframe in to type accepted by PyTorch
class DataFrameDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.data_tensor = torch.Tensor(df.values)

    # a function to get items by index
    def __getitem__(self, index):
        obj = self.data_tensor[index]
        input = self.data_tensor[index][0:-1]
        target = self.data_tensor[index][-1] - 1

        return input, target

    # a function to count samples
    def __len__(self):
        n, _ = self.data_tensor.shape
        return n
