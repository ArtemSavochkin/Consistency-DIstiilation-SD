import numpy as np
import torch
from torch import nn
from diffusers import StableDiffusionPipeline
from copy import deepcopy

DTYPE = torch.bfloat16


class ConsistencyLoss(nn.Module):
    def __init__(self):
        super(ConsistencyLoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, predict, target):
        return self.loss(predict, target)


def init_models(model_name, device):
    teacher_pipeline = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=DTYPE)
    student_unet = deepcopy(teacher_pipeline.unet)
    teacher_pipeline, student_unet = teacher_pipeline.to(device), student_unet.to(device)
    teacher_pipeline.vae.requires_grad_(False)
    teacher_pipeline.text_encoder.requires_grad_(False)
    teacher_pipeline.unet.requires_grad_(False)
    student_unet.train()
    return teacher_pipeline, student_unet


def predicted_origin(model_output,
                     timesteps,
                     boundary_timesteps,
                     sample,
                     prediction_type,
                     alphas,
                     sigmas,
                     pred_x_0=None):
    sigmas_s = extract_into_tensor(sigmas, boundary_timesteps, sample.shape)
    alphas_s = extract_into_tensor(alphas, boundary_timesteps, sample.shape)

    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)

    alphas_s[boundary_timesteps == 0] = 1.0
    sigmas_s[boundary_timesteps == 0] = 0.0

    if prediction_type == "epsilon":
        pred_x_0 = (sample - sigmas * model_output) / alphas if pred_x_0 is None else pred_x_0
        pred_x_0 = alphas_s * pred_x_0 + sigmas_s * model_output
    else:
        raise ValueError(f"Prediction type {prediction_type} currently not supported.")

    return pred_x_0


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DDIMSolver:
    def __init__(
            self, alpha_cumprods, timesteps=1000, ddim_timesteps=50,
            num_boundaries=1,
            num_inverse_boundaries=1,
            max_inverse_timestep_index=49
    ):
        step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (np.arange(1, ddim_timesteps + 1) * step_ratio).round().astype(
            np.int64) - 1  # [19, ..., 999]
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        self.ddim_alpha_cumprods_next = np.asarray(
            alpha_cumprods[self.ddim_timesteps[1:]].tolist() + [0.0]
        )

        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)
        self.ddim_alpha_cumprods_next = torch.from_numpy(self.ddim_alpha_cumprods_next)

        timestep_interval = ddim_timesteps // num_boundaries + int(ddim_timesteps % num_boundaries > 0)
        endpoint_idxs = torch.arange(timestep_interval, ddim_timesteps, timestep_interval) - 1
        self.endpoints = torch.tensor([0] + self.ddim_timesteps[endpoint_idxs].tolist())

        timestep_interval = ddim_timesteps // num_inverse_boundaries + int(ddim_timesteps % num_inverse_boundaries > 0)
        inverse_endpoint_idxs = torch.arange(timestep_interval, ddim_timesteps, timestep_interval) - 1
        inverse_endpoint_idxs = torch.tensor(inverse_endpoint_idxs.tolist() + [max_inverse_timestep_index])
        self.inverse_endpoints = self.ddim_timesteps[inverse_endpoint_idxs]

    def to(self, device):
        self.endpoints = self.endpoints.to(device)
        self.inverse_endpoints = self.inverse_endpoints.to(device)

        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        self.ddim_alpha_cumprods_next = self.ddim_alpha_cumprods_next.to(device)
        return self

    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_prev = extract_into_tensor(self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev
