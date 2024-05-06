

from torch.utils.data import DataLoader

import time
import datetime

from model import UNet, DiT
from ddpm import DDPM
from dataset import load_dataset


def main(epochs=10000, batch_size=32, lr=2e-4, T=1000, device="cuda"):
    log = open("logs/" + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M') + ".txt", "w")

    # model = UNet(T=T, ch=128, ch_mult=[1, 2, 3, 4], attn=[2], num_res_blocks=2).to(device)
    model = DiT(input_size=64, in_channels=3, hidden_size=512, depth=16).to(device)
    ddpm = DDPM(model=model, T=T, lr=lr, device=device)
    dataloader = DataLoader(load_dataset(device=device), batch_size=batch_size, shuffle=True, drop_last=True)

    start = time.time()
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            loss = ddpm.train_step(batch)

            # log to file
            log.write(str(loss) + "\n")
            log.flush()

            if step % 20 == 0:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss} Time: {time.time() - start}")

            if step % 500 == 0:
                ddpm.plot_denoising_process(save=f"images/{epoch}_{step}.png")


if __name__ == "__main__":
    main()

