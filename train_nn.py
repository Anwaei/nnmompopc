from datetime import datetime
import torch
import torch.utils
from torch.utils.data import Dataset, DataLoader
from torch import nn
from data_generate import OptimalDataset
from modules import OptimalModule
from torch.utils.tensorboard import SummaryWriter

def add_noise(x):
    x_a = x + torch.randn()

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for i_batch, sample_batched in enumerate(dataloader):
        x = sample_batched["input"].to(device)

        y = sample_batched["output"].to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i_batch % 100 == 0:
            loss_value, current = loss.item(), (i_batch + 1)*len(x)
            print(f"loss: {loss_value:>7f} [{current:>5d}/{size:>5d}]")
    return loss.item()

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss= 0
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            x = sample_batched["input"].to(device)
            y = sample_batched["output"].to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Avg loss: {test_loss:>8f} \n")
    return test_loss


if __name__ == "__main__":

    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
    print(f"Using {device} device")

    time_current = datetime.now().strftime('%m-%d-%H%M')

    writer = SummaryWriter('runs/run_'+time_current)

    dataset = torch.load('data/opt_data.pt')
    dataset_train, dataset_test = torch.utils.data.random_split(dataset, [0.8, 0.2])
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=20, shuffle=False)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=20, shuffle=False)
    # for i_batch, sample_batched in enumerate(dataloader):
    #     print(i_batch)
    #     print(sample_batched["input"][0, 0:8])
    #     print(sample_batched["output"][0, :])

    net = OptimalModule().to(device)
    print(net)

    # writer.add_graph(net)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.006, momentum=0.9)

    epoches = 500
    for t in range(epoches):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train(dataloader_train, net, loss_fn, optimizer)
        test_loss = test(dataloader_test, net, loss_fn)
        writer.add_scalars('loss', {'train':train_loss, 'test':test_loss}, t)
    
    save_path = 'model/net_' + time_current + '.pth'
    torch.save(net.state_dict(), save_path)
