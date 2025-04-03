import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import OneHotDegree


def load_proteins_dataset(batch_size=32, shuffle=True):
    """
    Loads the PROTEINS dataset (graph classification task) and returns 
    train/val/test dataloaders with fixed splits.
    
    Returns:
        train_loader, val_loader, test_loader (torch_geometric DataLoader)
    """

    dataset = TUDataset(root='data/PROTEINS', name='PROTEINS')

    # Shuffle and split
    dataset = dataset.shuffle()
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len

    train_dataset = dataset[:train_len]
    val_dataset = dataset[train_len:train_len + val_len]
    test_dataset = dataset[train_len + val_len:]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
