import torch


### Shared latent space losses

# from DreamFusion appendix A1: sds_loss(x) = weight(t) * dot(stopgrad[epshat_t - eps], x) where x = g(theta)
# and its grad is the known formula: weight(t) * (epshat_t - eps) * grad(x)
def get_sds_loss(z0_student, teacher, teacher_guidance_scale, eps, target_latent, timestep, w_t, eta_teacher, batch_s, batch_e, n_inv_iters=0, n_teacher_iters=1):
    with torch.inference_mode():
        if n_inv_iters > 0:
            z_t = teacher.noise_to_timestep_using_inversion(z0_student, teacher_guidance_scale, timestep.item(), n_inv_iters, batch_s, batch_e)
        else:
            z_t = teacher.noise_to_timestep(z0_student, timestep, eps)
       
        if target_latent is None:
            #e_t, pred_z0 = teacher.predict_eps_and_sample(z_t, timestep, teacher_guidance_scale, text_embeddings, batch_s, batch_e)
            e_t, pred_z0 = teacher.multi_step_sample(z_t, teacher_guidance_scale, timestep.item(), n_teacher_iters, batch_s, batch_e, eta_teacher)
        else: #TODO: move to teacher implementation!! this is not correct for all teachers
            pred_z0 = target_latent
            e_t = (z_t - teacher.alphas[timestep.item()] * pred_z0) / teacher.sigmas[timestep.item()]
        
        grad_z = w_t * (e_t - eps)
        grad_z = torch.nan_to_num(grad_z.detach(), 0.0, 0.0, 0.0)
    sds_loss = grad_z.clone() * z0_student
    sds_loss = sds_loss / z0_student.numel() # Daniel: normalization is necessary to avoid exploding grads
    del grad_z
    sds_loss = sds_loss.sum() 

    return sds_loss, z_t, pred_z0, z0_student



def get_image_mse_loss(z0_student, teacher, text_embeddings, teacher_guidance_scale, eps, timestep, w_t):
    with torch.inference_mode():
        z_t = teacher.noise_to_timestep(z0_student, timestep, eps)
        if text_embeddings is not None:
            e_t, pred_z0 = teacher.predict_eps_and_sample(z_t, timestep, teacher_guidance_scale, text_embeddings)
        else:
            e_t, pred_z0 = teacher.predict_eps_and_sample(z_t, timestep)

    mse = (((pred_z0.clone() - z0_student) ** 2) / pred_z0.numel()).sum()
    mse_loss = w_t * mse

    return mse_loss, z_t, pred_z0


### Non-shared latent space losses

def get_image_mse_pixels_loss(x0_student, teacher, text_embeddings, teacher_guidance_scale, eps, timestep, w_t, n_inv_iters=0):
    with torch.inference_mode():
        z0 = teacher.encode(x0_student, height=x0_student.shape[2], width=x0_student.shape[3])
        if n_inv_iters > 0:
            z_t = teacher.invert_to_timestep(z0, teacher_guidance_scale, text_embeddings, timestep, n_inv_iters)
        else:
            z_t = teacher.noise_to_timestep(z0, timestep, eps)
        
        if text_embeddings is not None:
            _, pred_z0 = teacher.predict_eps_and_sample(z_t, timestep, teacher_guidance_scale, text_embeddings)
        else:
            _, pred_z0 = teacher.predict_eps_and_sample(z_t, timestep)
                
        x0_teacher = teacher.decode(pred_z0, "pt", do_postprocess=False)
        #assert torch.isfinite(x0_teacher).all()


    mse = (((x0_teacher - x0_student) ** 2) / x0_teacher.numel()).sum()
    mse_loss = w_t * mse

    # Clean up unnecessary tensors
    del x0_teacher

    return mse_loss, z_t.detach(), pred_z0.detach(), z0.detach()


def get_image_mse_latent_loss(x0_student, teacher, text_embeddings, teacher_guidance_scale, eps, timestep, w_t, n_inv_iters=0):
    z0 = teacher.encode(x0_student, height=x0_student.shape[2], width=x0_student.shape[3])
    #z0 = torch.clamp(z0, -5.0, 5.0)
    with torch.inference_mode():
        if n_inv_iters > 0:
            z_t = teacher.invert_to_timestep(z0, teacher_guidance_scale, text_embeddings, timestep, n_inv_iters)
        else:
            z_t = teacher.noise_to_timestep(z0, timestep, eps)
        
        if text_embeddings is not None:
            _, pred_z0 = teacher.predict_eps_and_sample(z_t, timestep, teacher_guidance_scale, text_embeddings)
        else:
            _, pred_z0 = teacher.predict_eps_and_sample(z_t, timestep)

    mse = (((pred_z0 - z0) ** 2) / z0.numel()).sum()
    mse_loss = w_t * mse

    return mse_loss, z_t.detach(), pred_z0.detach(), z0.detach()


# def get_sds_pixels_loss(x0_student, teacher, text_embeddings, teacher_guidance_scale, eps, timestep, w_t, n_inv_iters=0):
#     with torch.inference_mode():
#         z0 = teacher.encode(x0_student, height=x0_student.shape[2], width=x0_student.shape[3])
#         if n_inv_iters > 0:
#             z_t = teacher.invert_to_timestep(z0, teacher_guidance_scale, text_embeddings, timestep, n_inv_iters)
#         else:
#             z_t = teacher.noise_to_timestep(z0, timestep, eps)
        
#         if text_embeddings is not None:
#             _, pred_z0 = teacher.predict_eps_and_sample(z_t, timestep, teacher_guidance_scale, text_embeddings)
#         else:
#             _, pred_z0 = teacher.predict_eps_and_sample(z_t, timestep)
                
#         x0_teacher = teacher.decode(pred_z0, "pt", do_postprocess=False)
#         assert torch.isfinite(x0_teacher).all()
    
#         grad_x = w_t * (x0_teacher - x0_student)
#         grad_x = torch.nan_to_num(grad_x.detach(), 0.0, 0.0, 0.0)

#     sds_loss = grad_x.clone() * x0_student
#     del grad_x
#     sds_loss = sds_loss / x0_student.numel() # Daniel: normalization is necessary to avoid exploding grads
#     sds_loss = sds_loss.sum() 
#     assert torch.isfinite(sds_loss).all()

    return sds_loss, z_t, pred_z0


def get_sds_latent_loss(x0_student, teacher, text_embeddings, teacher_guidance_scale, eps, timestep, w_t, batch_s, batch_e, n_inv_iters=0):
    z0 = teacher.encode(x0_student, height=x0_student.shape[2], width=x0_student.shape[3])
    with torch.inference_mode():
        if n_inv_iters > 0:
            z_t = teacher.invert_to_timestep(z0, teacher_guidance_scale, text_embeddings, timestep, n_inv_iters, batch_s, batch_e)
        else:
            z_t = teacher.noise_to_timestep(z0, timestep, eps)
        
        if text_embeddings is not None:
            e_t, pred_z0 = teacher.predict_eps_and_sample(z_t, timestep, teacher_guidance_scale, text_embeddings, batch_s, batch_e)
        else:
            e_t, pred_z0 = teacher.predict_eps_and_sample(z_t, timestep)
                
        assert torch.isfinite(pred_z0).all()
    
        grad_z = w_t * (e_t - eps)
        grad_z = torch.nan_to_num(grad_z.detach(), 0.0, 0.0, 0.0)

    sds_loss = grad_z.clone() * z0
    del grad_z
    sds_loss = sds_loss / z0.numel()  # Daniel: normalization is necessary to avoid exploding grads
    sds_loss = sds_loss.sum()

    return sds_loss, z_t, pred_z0, z0


def get_loss_f(loss_name):
    if loss_name == "sds_shared":
        return get_sds_loss
    # if loss_name == "sds_pixels":
    #     return get_sds_pixels_loss
    if loss_name == "sds_latent":
        return get_sds_latent_loss
    if loss_name == "mse":
        return get_image_mse_loss
    if loss_name == "mse_pixels":
        return get_image_mse_pixels_loss
    if loss_name == "mse_latent":
        return get_image_mse_latent_loss
    raise ValueError("unknown loss: {}".format(loss_name))
