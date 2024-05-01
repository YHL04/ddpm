

from torch.utils.data import DataLoader

import datetime

from unet import UNet
from ddpm import DDPM
from dataset import load_dataset


def main(epochs=10000, batch_size=64, lr=2e-4, T=500, device="cuda"):
    log = open("logs/" + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M') + ".txt", "w")

    model = UNet(T=T, ch=128, ch_mult=[1, 2, 3, 4], attn=[2], num_res_blocks=2).to(device)
    ddpm = DDPM(model=model, T=T, lr=lr, device=device)
    dataloader = DataLoader(load_dataset(device=device), batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            loss = ddpm.train_step(batch[0])

            # log to file
            log.write(str(loss) + "\n")
            log.flush()

            if step % 20 == 0:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss} ")

            if step % 300 == 0:
                ddpm.plot_denoising_process(save=f"images/{epoch}_{step}.png")


if __name__ == "__main__":
    main()

