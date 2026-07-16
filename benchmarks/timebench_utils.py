import time
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Subset

from hyrax import Hyrax

# DATA_PATH = "/astro/store/shiren/wqy37/datasets"

# hyrax bench 
def benchmark_hyrax(train_fraction=1.0, epochs=10, batch_size=512, num_workers=0, lr=0.01):
    # setup
    h = Hyrax()
    h.set_config("model.name", "HyraxCNN")
    h.set_config("data_loader.batch_size", batch_size)
    h.set_config("train.epochs", epochs)
    h.set_config("data_loader.num_workers", num_workers)

    # Match the plain PyTorch optimizer settings so the comparison is more fair.
    h.config["torch.optim.SGD"] = {"lr": lr, "momentum": 0.9}

    data_request_definition = {
        "train": {
            "data": {
                "dataset_class": "HyraxCifarDataset",
                "data_location": "./data",
                "fields": ["image", "label"],
                "primary_id_field": "object_id",
            },
        },
        "infer": {
            "data": {
                "dataset_class": "HyraxCifarDataset",
                "data_location": "./data",
                "fields": ["image"],
                "primary_id_field": "object_id",
                "dataset_config": {
                    "HyraxCifarDataset": {
                        "use_training_data": False,
                    },
                },
            },
        },
    }
    h.set_config("data_request", data_request_definition)

    split = {
        "train": train_fraction
    }
    h.set_config("split", split)

    # train
    start_train = time.perf_counter()
    trained_model = h.train()
    end_train = time.perf_counter()

    # infer
    start_infer = time.perf_counter()
    inference_results = h.infer()
    end_infer = time.perf_counter()

    # accuracy
    predicted_classes = np.zeros(len(inference_results)).astype(int)
    for i, result in enumerate(inference_results):
        predicted_classes[i] = np.argmax(result)

    with open("./data/cifar-10-batches-py/test_batch", "rb") as f_in:
        test_data = pickle.load(f_in, encoding="bytes")

    y_true = test_data[b"labels"]
    y_pred = predicted_classes.tolist()
    correct = 0
    for t, p in zip(y_true, y_pred):
        correct += t == p
    accuracy = correct / len(y_true)

    print(
        f"Hyrax | fraction={train_fraction:.2f} | epochs={epochs} | batch_size={batch_size} | "
        f"lr={lr:.3f} | num_workers={num_workers} | "
        f"train={end_train - start_train:.3f}s | infer={end_infer - start_infer:.3f}s | "
        f"accuracy={accuracy:.4f}"
    )

    return {
        "framework": "hyrax",
        "train_fraction": train_fraction,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "lr": lr,
        "train_time": end_train - start_train,
        "infer_time": end_infer - start_infer,
        "num_infer_samples": len(inference_results),
        "accuracy": accuracy,
    }


# pytorch bench 
def build_cifar_loader(train_fraction=1.0, batch_size=512, num_workers=0):
    # Match the hyrax implementation
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    n_train = int(len(train_dataset) * train_fraction)
    torch.manual_seed(0)
    indices = torch.randperm(len(train_dataset))[:n_train]
    train_subset = Subset(train_dataset, indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers)

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                             shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def benchmark_pytorch(train_fraction=1.0, epochs=10, batch_size=512, num_workers=0, lr=0.01):
    train_loader, test_loader = build_cifar_loader(train_fraction=train_fraction, batch_size=batch_size, num_workers=num_workers)
    net = Net()
    device = next(net.parameters()).device
    device = torch.device(torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu')
    net.to(device)

    print("Model running on:", next(net.parameters()).device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    # train
    start_train = time.perf_counter()
    for epoch in range(epochs):
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    end_train = time.perf_counter()

    # infer
    start_infer = time.perf_counter()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    end_infer = time.perf_counter()

    accuracy = correct / total
    print(
        f"PyTorch | fraction={train_fraction:.2f} | epochs={epochs} | batch_size={batch_size} | "
        f"lr={lr:.3f} | num_workers={num_workers} | "
        f"train={end_train - start_train:.3f}s | infer={end_infer - start_infer:.3f}s | "
        f"accuracy={accuracy:.3f}"
    )

    return {
        "framework": "pytorch",
        "train_fraction": train_fraction,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "lr": lr,
        "train_time": end_train - start_train,
        "infer_time": end_infer - start_infer,
        "num_infer_samples": total,
        "accuracy": accuracy,
    }

# trail function
def benchmark_repeated(framework, train_fraction=1.0, epochs=10, batch_size=512, num_workers=0, lr=0.01, repeats=3):
    if framework == "hyrax":
        runner = benchmark_hyrax
    elif framework == "pytorch":
        runner = benchmark_pytorch
    else:
        raise ValueError("framework must be 'hyrax' or 'pytorch'")

    results = []
    for i in range(repeats):
        print(f"{framework} repeat {i + 1}/{repeats}")
        results.append(
            runner(
                train_fraction=train_fraction,
                epochs=epochs,
                num_workers=num_workers,
                batch_size=batch_size,
                lr=lr,
            )
        )

    train_times = [r["train_time"] for r in results]
    infer_times = [r["infer_time"] for r in results]
    accuracies = [r["accuracy"] for r in results]

    summary = {
        "framework": framework,
        "train_fraction": train_fraction,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "repeats": repeats,
        "train_time_mean": float(np.mean(train_times)),
        "train_time_std": float(np.std(train_times)),
        "infer_time_mean": float(np.mean(infer_times)),
        "infer_time_std": float(np.std(infer_times)),
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
    }

    print(
        f"{framework.upper()} avg over {repeats} runs | "
        f"train={summary['train_time_mean']:.3f}s ± {summary['train_time_std']:.3f}s | "
        f"infer={summary['infer_time_mean']:.3f}s ± {summary['infer_time_std']:.3f}s | "
        f"accuracy={summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}"
    )

    return summary
