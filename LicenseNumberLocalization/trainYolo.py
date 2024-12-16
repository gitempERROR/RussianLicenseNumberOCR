import torchinfo

import configYolo

from modelYoloV3 import YoloV3
from tqdm import tqdm
from utilsYolo import *
from lossYolo import YoloLoss


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    """
    :param train_loader: Загрузчик набора данных
    :param model: Модель
    :param optimizer: Оптимизатор
    :param loss_fn: Функция потерь
    :param scaler: Скейлер для экономия видеопамяти
    :param scaled_anchors: Якорные точки приведенные к размеру масштаба их рамки
    :return:
    """
    loop = tqdm(train_loader, leave=True)
    losses = []

    for x, y in loop:
        x = x.to(configYolo.DEVICE)

        y0, y1, y2 = (
            y[0].to(configYolo.DEVICE),
            y[1].to(configYolo.DEVICE),
            y[2].to(configYolo.DEVICE)
        )

        if configYolo.DEVICE == 'cuda':
            with torch.cuda.amp.autocast():
                out = model(x)

                loss = (
                    loss_fn(out[0], y0, scaled_anchors[0])
                    + loss_fn(out[1], y1, scaled_anchors[1])
                    + loss_fn(out[2], y2, scaled_anchors[2])
                )
            losses.append(loss.item())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            mean_loss = sum(losses) / len(losses)
            loop.set_postfix(loss=mean_loss)

        elif configYolo.DEVICE == 'cpu':
            out = model(x)

            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mean_loss = sum(losses) / len(losses)
            loop.set_postfix(loss=mean_loss)


def main():
    model = YoloV3().to(configYolo.DEVICE)
    torchinfo.summary(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=configYolo.LEARNING_RATE, weight_decay=configYolo.WEIGHT_DECAY)
    loss_fn = YoloLoss()
    train_loader = get_loaders()
    scaled_anchors = anchor_scaler()

    if configYolo.DEVICE == 'cuda':
        scaler = torch.cuda.amp.GradScaler()

    if configYolo.LOAD_MODEL:
        load_checkpoint(model, optimizer)

    for epoch in range(1, configYolo.NUM_EPOCHS+1):
        print(f'Epoch {epoch}/{configYolo.NUM_EPOCHS}')
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

        if configYolo.SAVE_MODEL:
            save_checkpoint(model, optimizer)


if __name__ == '__main__':
    main()
