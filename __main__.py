

from torch.utils.data import DataLoader

from ddpm import DDPM
from dataset import load_dataset


def main(epochs=10000, batch_size=128):

    ddpm = DDPM(T=500, device="cuda")

    data = load_dataset()
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            loss = ddpm.train_step(batch[0])

            if step % 20 == 0:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss} ")

            if step % 100 == 0:
                ddpm.plot_denoising_process(save=f"images/{epoch}_{step}.png")


if __name__ == "__main__":
    main()

