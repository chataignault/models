import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.nn import CrossEntropyLoss


def train_model(
    fm,
    nepochs: int,
    learning_rate: float,
    train_loader,
    test_loader,
    alpha1: float = 0.0,
    alpha2: float = 0.0,
):
    fm.train()
    optimizer = optim.Adam(fm.parameters(), lr=learning_rate)

    # lin_scheduler = optim.lr_scheduler.LinearLR(
    #     optimizer=optimizer,
    #     start_factor=1 / 10,
    #     end_factor=1.0,
    #     total_iters=5,
    # )
    # cos_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer=optimizer,
    #     T_0=10,
    #     eta_min=2e-5,
    # )
    # exp_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)
    # scheduler = optim.lr_scheduler.SequentialLR(
    #     optimizer=optimizer,
    #     schedulers=[lin_scheduler, cos_scheduler, exp_scheduler],
    #     milestones=[5, 25],
    # )
    lr = learning_rate

    lossx, lossx_test, gradientx, lrx, param_changes = [], [], [], [], []
    loss_function = CrossEntropyLoss()
    for epoch in tqdm(range(nepochs)):
        params = [torch.clone(p) for p in fm.parameters()]
        for x, y in train_loader:
            optimizer.zero_grad()
            y_hat = fm(x)
            # loss = torch.sum((y - y_hat) ** 2) / len(y)
            loss = loss_function(y_hat, torch.softmax(y, axis=1))
            if (alpha1 > 0.0) or (alpha2 > 0.0):
                # sx = fm.forward_partial(x)
                # n_lie_params = sum([np.prod(l.size()) for l in sx])
                if alpha1 > 0.0:
                    dg = fm.atdev.development_layers.parameters()

                    # loss += (
                    #     alpha1
                    #     * sum(
                    #         [
                    #             torch.linalg.norm(
                    #                 torch.linalg.norm(l, ord=1, dim=(2, 3)),
                    #                 ord=1,
                    #                 dim=(0, 1),
                    #             )
                    #             for l in sx
                    #         ]
                    #     )
                    #     / n_lie_params
                    #     / len(y)
                    # )
                    loss += (
                        alpha1 * 
                        sum(
                            torch.sum(
                                torch.abs(d)
                            ) / np.prod(list(d.shape))
                            for d in dg
                        )
                    )
                # if alpha2 > 0.0:
                #     loss += (
                #         alpha2
                #         * sum([torch.linalg.norm(s) for s in sx])
                #         / n_lie_params
                #         / len(y)
                #     )

            loss.backward()
            lossx.append(loss.item())
            optimizer.step()
        # lr = scheduler.get_last_lr()[0]
        lrx.append(lr)
        gradientx.append(
            np.mean(
                [
                    torch.linalg.norm(p.grad).detach().cpu().numpy()
                    for p in fm.parameters()
                ]
            )
        )
        # scheduler.step()
        print(
            f"Epoch : {epoch:<2} | "
            f"Loss {np.round(lossx[-1], decimals=2):<5} | "
            f"Gradient {np.round(float(gradientx[-1]), decimals=4):<5} | "
            f"Lr {np.round(lr, decimals=5)}"
        )
        new_params = list(fm.parameters())
        param_changes.append(
            [
                (torch.linalg.norm(p - pn) / torch.linalg.norm(p))
                .detach()
                .cpu()
                .numpy()
                for (p, pn) in zip(params, new_params)
            ]
        )
        if epoch % 4 != 0:
            continue
        # compute the test error
        fm.eval()
        for x, y in test_loader:
            y_hat = fm(x)
            # loss = torch.sum((y - y_hat) ** 2) / len(y)
            loss = loss_function(y_hat, torch.softmax(y, axis=1))
            lossx_test.append(loss.item())
        fm.train()

    return fm, lossx, lossx_test, gradientx, lrx, param_changes


def plot_average_development(fm, n_samples, tsx_train, y_train_labels):
    for label_class in range(6):
        tsx_train_class = tsx_train[
            y_train_labels.astype(float).astype(int) == label_class
        ]
        sx = fm.forward_partial(tsx_train_class[:n_samples])
        k = len(sx)
        fig, axs = plt.subplots(
            ncols=k, nrows=2, figsize=(10, 4), sharex=True, sharey=True
        )
        for i in range(k):
            axs[0, i].imshow(torch.mean(sx[i], axis=0).cpu().detach().numpy()[0, :, :])
            axs[0, i].set_title(
                f"Norm : {np.round(torch.mean(torch.linalg.norm(sx[i], dim=(1, 2))).item(), decimals=2)}"
            )
            im = axs[1, i].imshow(
                torch.mean((sx[i] - torch.mean(sx[i], axis=0)) ** 2, axis=0)
                .cpu()
                .detach()
                .numpy()[0, :, :],
                cmap="magma",
                vmin=0.0,
                vmax=0.4,
            )

        plt.colorbar(im)
        plt.tight_layout()
        fig.suptitle(f"Embedding for class {label_class}")
