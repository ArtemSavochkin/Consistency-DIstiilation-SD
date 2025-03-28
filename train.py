import os
import torch
from diffusers import DDPMScheduler
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import get_dataloader
from utils import init_models, ConsistencyLoss, DDIMSolver, predicted_origin, DTYPE


def get_text_embeddings(pipeline, text, device):
    tokens = pipeline.tokenizer(text,
                                padding="max_length",
                                max_length=pipeline.tokenizer.model_max_length,
                                truncation=True,
                                return_tensors="pt")['input_ids']
    tokens = tokens.to(device)
    embeddings = pipeline.text_encoder(tokens)[0]
    return embeddings


def train_one_epoch(loader, teacher_pipe, student_unet, device, optimizer, noise_scheduler, solver,
                    empty_text_embeddings, loss_function, num_ddim_timesteps=50, guidance_scale=8
                    ):
    epoch_loss = 0
    for i, batch in enumerate(tqdm(loader, desc="Epoch_train")):
        loss = process_batch(batch, device, teacher_pipe, student_unet, optimizer, noise_scheduler, solver,
                             empty_text_embeddings, loss_function, num_ddim_timesteps, guidance_scale)
        epoch_loss += loss.item()
    else:
        epoch_loss /= i
    return epoch_loss


def process_batch(batch, device, teacher_pipe, student_unet, optimizer, noise_scheduler, solver, empty_text_embeddings,
                  loss_function,
                  num_ddim_timesteps=50, guidance_scale=8):
    # переводим батч на гпу
    images = batch["image"]
    text = batch["prompt"]
    images = images.to(device)
    images = images.to(dtype=DTYPE)
    # создаем эмбединги текстовых запросов
    text_embeddings = get_text_embeddings(teacher_pipe, text, device)
    # прогоняем изображения через энкодер
    encoded_image = teacher_pipe.vae.encode(images).latent_dist.sample() * teacher_pipe.vae.config.scaling_factor

    # сэмплируем данные
    index = torch.randint(0, num_ddim_timesteps, (encoded_image.shape[0],), device=device).long()
    topk = noise_scheduler.config.num_train_timesteps // num_ddim_timesteps
    start_timesteps = solver.ddim_timesteps[index]
    timesteps = torch.clamp(start_timesteps - topk, 0, solver.ddim_timesteps[-1])
    alpha = torch.sqrt(noise_scheduler.alphas_cumprod)
    alpha = alpha.to(device)
    beta = torch.sqrt(1 - noise_scheduler.alphas_cumprod)
    beta = beta.to(device)

    noise = torch.randn(encoded_image.shape).to(device)
    noise = noise.to(dtype=DTYPE)
    # зашумляем картинку
    noised_encoded_image = noise_scheduler.add_noise(encoded_image, noise, start_timesteps)

    # предсказание сети-студента из x_n

    # шум
    student_noise_pred = student_unet(
        noised_encoded_image,
        start_timesteps,
        encoder_hidden_states=text_embeddings,
    ).sample

    # картинка
    x_pred_student = predicted_origin(
        student_noise_pred,
        start_timesteps,
        torch.zeros_like(start_timesteps),
        noised_encoded_image,
        noise_scheduler.config.prediction_type,
        alpha,
        beta,
    )
    # предсказание сети-учителя

    # шум с текстом и без текста
    with torch.no_grad():
        teacher_noise_pred_cond = teacher_pipe.unet(
            noised_encoded_image,
            start_timesteps,
            encoder_hidden_states=text_embeddings
        ).sample
        teacher_noise_pred_uncond = teacher_pipe.unet(
            noised_encoded_image,
            start_timesteps,
            encoder_hidden_states=empty_text_embeddings
        ).sample
        teacher_noise_pred = teacher_noise_pred_uncond + guidance_scale * (
                teacher_noise_pred_cond - teacher_noise_pred_uncond)

        # картинка с текстом и без текста
        teacher_x0_cond = predicted_origin(
            teacher_noise_pred_cond,
            start_timesteps,
            torch.zeros_like(start_timesteps),
            noised_encoded_image,
            noise_scheduler.config.prediction_type,
            alpha,
            beta,
        )
        teacher_x0_uncond = predicted_origin(
            teacher_noise_pred_uncond,
            start_timesteps,
            torch.zeros_like(start_timesteps),
            noised_encoded_image,
            noise_scheduler.config.prediction_type,
            alpha,
            beta,
        )
        teacher_x0 = teacher_x0_cond + guidance_scale * (teacher_x0_cond - teacher_x0_uncond)
        previous_x = solver.ddim_step(teacher_x0, teacher_noise_pred, index)
        previous_x = previous_x.to(dtype=DTYPE)

        # предсказания сети-студента от previous_x
        pred_noise_prev_x = student_unet(
            previous_x,
            timesteps,
            encoder_hidden_states=empty_text_embeddings
        ).sample
        pred_x = predicted_origin(
            pred_noise_prev_x,
            timesteps,
            torch.zeros_like(start_timesteps),
            previous_x,
            noise_scheduler.config.prediction_type,
            alpha,
            beta,
        )
    loss = loss_function(x_pred_student, pred_x)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student_unet.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss


def train(args):
    device = args.device
    epochs = args.num_epochs
    guidance_scale = args.guidance_scale
    checkpoint_frequency = args.checkpoint_frequency
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    # модели
    teacher_pipeline, student_unet = init_models(model_name=args.model_name, device=device)
    # даталоадер

    dataloader = get_dataloader(args.path_to_dataset, args.data_file_name, batch_size=args.batch_size)
    # оптимизатор
    optimizer = torch.optim.AdamW(student_unet.parameters(), lr=float(args.learning_rate))
    # Объект, осуществляющий зашумление
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_name, subfolder="scheduler")
    # Солвер
    solver = DDIMSolver(noise_scheduler.alphas_cumprod)
    solver = solver.to(device)

    loss_per_epochs = []
    epochs_list = []
    # функция потерь
    loss_function = ConsistencyLoss()
    # текстовые эмбединнги для безусловного расшумления
    empty_text_embeddings = get_text_embeddings(teacher_pipeline, [""] * args.batch_size, device)

    for epoch in tqdm(range(1, epochs + 1), "Training:"):
        epoch_loss = train_one_epoch(dataloader, teacher_pipeline, student_unet, device, optimizer, noise_scheduler,
                                     solver, empty_text_embeddings, loss_function, 50, guidance_scale)
        print(f"epoch_loss: {epoch_loss}")
        loss_per_epochs.append(epoch_loss)
        epochs_list.append(epoch)
        if epoch % checkpoint_frequency == 0 or epoch == 1 or epoch == epochs:
            torch.save(student_unet.state_dict(), os.path.join(checkpoint_dir, f"{epoch}_model.pth"))

    plt.plot(epochs_list, loss_per_epochs)
    plt.xlabel("Epochs")
    plt.ylabel("Training loss")
    plt.savefig("Training.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_dataset", required=False,
                        default="/workspace/fundament/model-f/imagetotext/deepseek/sd/dataset")  # True
    parser.add_argument("--data_file_name", required=False, default="valid_anno_repath.jsonl")  # True
    parser.add_argument("--model_name", required=False, default="sd-legacy/stable-diffusion-v1-5")
    parser.add_argument("--batch_size", "-b", required=False, default=16)
    parser.add_argument("--device", required=False, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_epochs", required=False, default=10)
    parser.add_argument("--learning_rate", required=False, default=0.001)
    parser.add_argument("--guidance_scale", required=False, default=8)
    parser.add_argument("--checkpoint_frequency", required=False, default=1)
    parser.add_argument("--checkpoint_dir", required=False, default="./checkpoint/")
    arguments = parser.parse_args()
    train(arguments)
