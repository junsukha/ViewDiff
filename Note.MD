Train 할 때는 3.4 Implementation Details의 Training에서처럼 다음과 같이 Unconditional Generation과 Image-Conditional Generation을 같이 사용함 &#8595;
```python
# train.py

# check if some timesteps within the batch will be replaced with 0 (== non-noisy image)
if not is_dreambooth and finetune_config.training.prob_images_not_noisy > 0:
    random_p_non_noisy = torch.rand((batch_size, finetune_config.model.n_input_images), device=latents.device, generator=generator) # n_iput_images is the number of images 
    non_noisy_mask = random_p_non_noisy < finetune_config.training.prob_images_not_noisy
    non_noisy_mask[:, finetune_config.training.max_num_images_not_noisy:] = False
    non_noisy_mask = non_noisy_mask.flatten()
    print(f'non_noisy_mask: {non_noisy_mask}')
    timesteps = torch.where(non_noisy_mask, torch.zeros_like(timesteps), timesteps) 
```
\
\
하지만 test할 때는 Diffusion pipeline 내에서 input (condition) 으로 쓰일 이미지들 갯수가 n_known_images &#8595;

```python
# custom_stable_diffusion_pipeline.py

# 5.a check if known_images are provided as input
# timestep will be set to 0 for known images in self.unet forward function
n_known_images = 0
if known_images is not None:
    n_known_images = known_images.shape[0]
    known_image_latents = self._encode_vae_image(image=known_images, generator=generator)
    latents[:n_known_images] = known_image_latents
```
\
\
위에서 define된 latents 가 unet 내에서 denoising step을 거칠 때 condition으로 쓰인 images에 해당하는 t 는 0으로 세팅 &#8595;
```python
# custom_unet_2d_condition.py

if n_known_images > 0: # I think this is the case for test step
    timesteps = timesteps.clone()
    timesteps = expand_batch(timesteps, n_images_per_batch)
    timesteps[:, 0:n_known_images] = 0
    timesteps = collapse_batch(timesteps)
    #print(f"set timestep 0 for n_known_images={n_known_images} -->", timesteps, timesteps.shape)
```
\
\
latents 의 batch size가 3이니깐. 이미지 3장을 가져와서 (dataset에서?). encode한게 latents. 그 세장중에서 0.25확률로 첫번째 그리고(또는) 두번째 timestep을 0으로. 즉 input이미지를 그대로 사용 (without noising). 나머지 이미지는 diffusion (add noise)해서 train에 사용. 그럼 질문은. non noise이미지를 어떻게 사용? denoising할때 condition image로 사용? 그럼 첫 두 이미지(or latents)에서 timestep 이 0이 아닌 경우 (noise된 경우는) unconditional의미? 첫 두개다 timestep이 0이 아니어야 unconditional인가? 둘 중 하나만 timestep이 0인경우는? 둘다 0인 경우는?  custom_unet_2d_condition에서 어떻게 사용되는지 보기. batch는 각각의 샘플이 독립적으로 훈련되는거임. 그걸 여러개 한꺼번에 하는게 batch일뿐. 그럼 이미지3장이 각각 timestep이 어떤지는 서로 관련이 없음 (다만 noise한다면 같은 timestep사용하는듯.) 관련이 없다는건 서로 뭔가를 공유하거나 하지 않는다는 듯. 그럼 왜 첫 두장만 0.25확률로 하는거지? 세번째는? 네번째는? 아 0.25확률로 첫번째랑 두번째 각각 image-conditional로 훈련하고 (두 개다 0.25확률에 당첨됏을때도 그냥 단순히 각각 image-conditonal로 훈련되는거임. 서로 영향주는거없는듯) 나머지 0.75 확률로는 unconditional로 대부분 훈련하는 경우. 세번째 이후 정확히는 (max_num_images_not_noisy 이후)의 image 들의 경우는 항상 false로 noise더해줌 

\
\
