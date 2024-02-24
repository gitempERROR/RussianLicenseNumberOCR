from torchinfo import summary
from tqdm import tqdm
from simpleOCR import SimpleOCR
from datasetOCR import OCRDataset
from utilsOCR import *


def train_loop(model, optimizer, loss_fn, scaler, train_loader):
    loop = tqdm(train_loader, leave=True)
    losses = []

    for image, label, label_string in loop:
        image = image.to(configOCR.DEVICE)
        label = label.to(configOCR.DEVICE)

        if configOCR.DEVICE == 'cuda':
            with torch.cuda.amp.autocast():
                prediction = model(image)
                bool_tensor = prediction[..., 0] < 25
                loss = loss_fn(prediction[..., 0:][bool_tensor], label[..., 0][bool_tensor].long())

            losses.append(loss.item())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            prediction = model(image)
            loss = loss_fn(prediction[..., 0:], label[..., 0])

            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)


def main():
    model = SimpleOCR().to(configOCR.DEVICE)
    summary(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=configOCR.LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()
    dataset = OCRDataset(configOCR.IMAGE_DIR, configOCR.TRANSFORMS)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        num_workers=configOCR.NUM_WORKERS,
        shuffle=True,
        batch_size=configOCR.BATCH_SIZE,
        drop_last=True
    )

    if configOCR.DEVICE == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    if configOCR.LOAD_MODEL:
        load_model_checkpoint(model, optimizer)

    for epoch in range(configOCR.NUM_EPOCHS):
        print(rf"epoch {epoch+1}/{configOCR.NUM_EPOCHS}")
        train_loop(model, optimizer, loss_fn, scaler, dataloader)
        if configOCR.SAVE_MODEL:
            save_model_checkpoint(model, optimizer, configOCR.MODEL_DIR)