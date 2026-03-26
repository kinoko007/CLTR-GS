import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from glob import glob
from transformers import CLIPTokenizer

from See3D_modules.mv_diffusion import mvdream_diffusion_model
from See3D_modules.mv_diffusion_SR import mvdream_diffusion_model as mvdream_diffusion_model_SR

from argparse import ArgumentParser
import matplotlib.pyplot as plt
import gc
import time


class See3D(nn.Module):
    def __init__(
        self,
        device,
        base_model_path='./checkpoint/MVD_weights/',
        model_type='sparse',  # single or sparse
        use_SR=False,
        seed=12345,
    ):
        super().__init__()

        self.device = device
        mv_unet_path = os.path.join(base_model_path, f"unet/{model_type}/ema-checkpoint")

        tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
        self.rgb_model = mvdream_diffusion_model(base_model_path, mv_unet_path, tokenizer, seed=seed)
        if use_SR:
            self.rgb_model_SR = mvdream_diffusion_model_SR(base_model_path, mv_unet_path, tokenizer, seed=seed)

    def PIL2tensor(self, height, width, num_frames, masks, warps, logicalNot=False):
        channels = 3
        condition_pixel_values = torch.empty((num_frames, channels, height, width))
        masks_pixel_values = torch.ones((num_frames, 1, height, width))

        prompt = ''

        # masks -> [0,1]
        for i, img in enumerate(masks):
            img = img.convert('L')
            img_resized = img.resize((width, height))
            img_tensor = torch.from_numpy(np.array(img_resized)).float()
            img_normalized = img_tensor / 255.0
            mask_condition = (img_normalized > 0.9).float()
            masks_pixel_values[i] = mask_condition

        # warps -> [-1,1]
        for i, img in enumerate(warps):
            img_resized = img.resize((width, height))
            img_tensor = torch.from_numpy(np.array(img_resized)).float()
            img_normalized = img_tensor / 127.5 - 1.0
            img_normalized = img_normalized.permute(2, 0, 1)

            if logicalNot:
                img_normalized = torch.logical_not(masks_pixel_values[i]) * (-1) + masks_pixel_values[i] * img_normalized
            condition_pixel_values[i] = img_normalized

        return [prompt], {
            'conditioning_pixel_values': condition_pixel_values,  # [-1,1]
            'masks': masks_pixel_values  # [0,1]
        }

    def get_image_files(self, folder_path):
        image_extensions = [
            '*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp',
            '*.JPG', '*.JPEG', '*.PNG', '*.GIF', '*.BMP', '*.TIFF', '*.WEBP'
        ]
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob(os.path.join(folder_path, ext)))
        image_names = [os.path.basename(file) for file in image_files]
        image_names.sort()
        return image_names

    def load_ref_images(self, folder_path, height_mvd, width_mvd):
        temp_image_names = self.get_image_files(folder_path)
        if len(temp_image_names) == 0:
            raise RuntimeError(f"[see3d] No reference images in: {folder_path}")

        temp_img_path = os.path.join(folder_path, temp_image_names[0])
        temp_img = Image.open(temp_img_path).convert("RGB")
        width_ref, height_ref = temp_img.size

        ref_images = []
        ref_image_names = []

        # resize / split if needed
        if height_ref != height_mvd or width_ref != width_mvd:
            if height_ref > width_ref:
                height_tgt = int(height_ref * width_mvd / max(width_ref, 1))
                for image_name in temp_image_names:
                    img = Image.open(os.path.join(folder_path, image_name)).convert("RGB")
                    img = img.resize((width_mvd, height_tgt), resample=Image.BILINEAR)

                    img_top = img.crop((0, 0, width_mvd, height_mvd))
                    img_bottom = img.crop((0, max(height_tgt - height_mvd, 0), width_mvd, height_tgt))

                    img_name_top = image_name.split('.')[0] + '_top.png'
                    img_name_bottom = image_name.split('.')[0] + '_bottom.png'

                    ref_images.append(img_top)
                    ref_images.append(img_bottom)
                    ref_image_names.append(img_name_top)
                    ref_image_names.append(img_name_bottom)

            elif width_ref > height_ref:
                width_tgt = int(width_ref * height_mvd / max(height_ref, 1))
                for image_name in temp_image_names:
                    img = Image.open(os.path.join(folder_path, image_name)).convert("RGB")
                    img = img.resize((width_tgt, height_mvd), resample=Image.BILINEAR)

                    img_left = img.crop((0, 0, width_mvd, height_mvd))
                    img_right = img.crop((max(width_tgt - width_mvd, 0), 0, width_tgt, height_mvd))

                    img_name_left = image_name.split('.')[0] + '_left.png'
                    img_name_right = image_name.split('.')[0] + '_right.png'

                    ref_images.append(img_left)
                    ref_images.append(img_right)
                    ref_image_names.append(img_name_left)
                    ref_image_names.append(img_name_right)

            else:
                for image_name in temp_image_names:
                    img = Image.open(os.path.join(folder_path, image_name)).convert("RGB")
                    img = img.resize((width_mvd, height_mvd), resample=Image.BILINEAR)
                    ref_images.append(img)
                    ref_image_names.append(image_name)
        else:
            for image_name in temp_image_names:
                img = Image.open(os.path.join(folder_path, image_name)).convert("RGB")
                ref_images.append(img)
                ref_image_names.append(image_name)

        if len(ref_images) == 0:
            raise RuntimeError("[see3d] load_ref_images produced 0 ref images (unexpected).")

        return ref_images, ref_image_names

    def _collect_warp_names(self, warp_root_dir):
        names = sorted([os.path.basename(p) for p in glob(os.path.join(warp_root_dir, "warp_frame*.png"))])
        if len(names) == 0:
            names = sorted([os.path.basename(p) for p in glob(os.path.join(warp_root_dir, "warp_*.png"))])
        if len(names) == 0:
            raise RuntimeError(f"[see3d] No warp images found in: {warp_root_dir}")
        return names

    def _mask_name_from_warp(self, warp_name):
        if "warp_frame" in warp_name:
            return warp_name.replace("warp_frame", "mask_frame")
        return warp_name.replace("warp_", "mask_")

    def inpainting(
        self,
        source_imgs_dir,
        warp_root_dir,
        output_root_dir,
        super_resolution=False,
        batch_warp_num=10,
        carry_over=True,
    ):
        os.makedirs(output_root_dir, exist_ok=True)

        height_mvd = 512
        width_mvd = 512

        ref_images, ref_image_names = self.load_ref_images(source_imgs_dir, height_mvd, width_mvd)
        gt_num_b = len(ref_images)

        full_mask = np.ones((height_mvd, width_mvd), dtype=np.uint8) * 255
        full_mask_img = Image.fromarray(full_mask).convert("L")
        ref_masks = [full_mask_img for _ in range(gt_num_b)]
        ref_warps = [im.convert("RGB") for im in ref_images]

        if warp_root_dir is None or (not os.path.isdir(warp_root_dir)):
            raise NotADirectoryError(f"[see3d] warp_root_dir is not a directory: {warp_root_dir}")

        warp_names = self._collect_warp_names(warp_root_dir)

        # output size
        fimage = Image.open(os.path.join(warp_root_dir, warp_names[0])).convert("RGB")
        (width_out, height_out) = fimage.size

        print(f"[see3d] refs={gt_num_b}, warps={len(warp_names)}, batch_warp_num={batch_warp_num}, carry_over={carry_over}")

        images_predict = []          # keep for SR stage
        images_predict_names = []    # keep for SR stage

        prev_pred = None

        for start in range(0, len(warp_names), batch_warp_num):
            chunk_names = warp_names[start:start + batch_warp_num]

            chunk_warps, chunk_masks = [], []
            for wn in chunk_names:
                warp_path = os.path.join(warp_root_dir, wn)
                mask_path = os.path.join(warp_root_dir, self._mask_name_from_warp(wn))
                if not os.path.exists(mask_path):
                    raise FileNotFoundError(f"[see3d] Missing mask: {mask_path}")
                chunk_warps.append(Image.open(warp_path).convert("RGB"))
                chunk_masks.append(Image.open(mask_path).convert("L"))

            use_ctx = carry_over and (prev_pred is not None)

            # batch = refs + (optional ctx) + chunk
            masks_infer_batch = list(ref_masks)
            warp_infer_batch = list(ref_warps)

            if use_ctx:
                masks_infer_batch.append(full_mask_img)
                warp_infer_batch.append(prev_pred.convert("RGB"))

            ctx_offset = 1 if use_ctx else 0
            gt_num_frames = gt_num_b + ctx_offset

            masks_infer_batch += chunk_masks
            warp_infer_batch += chunk_warps

            prompt, batch = self.PIL2tensor(
                height_mvd, width_mvd,
                len(masks_infer_batch),
                masks_infer_batch, warp_infer_batch,
                logicalNot=False
            )

            images_predict_batch = self.rgb_model.inference_next_frame(
                prompt, batch,
                len(masks_infer_batch),
                height_mvd, width_mvd,
                gt_num_frames=gt_num_frames,
                output_type='pil'
            )

            pred_chunk = images_predict_batch[gt_num_frames: gt_num_frames + len(chunk_names)]

            for img_pred, wn in zip(pred_chunk, chunk_names):
                img_pred.resize((width_out, height_out)).save(os.path.join(output_root_dir, f"predict_{wn}"))
                images_predict.append(img_pred)          # for SR
                images_predict_names.append(wn)          # for SR

            prev_pred = pred_chunk[-1]  # carry to next batch

            print(f"[see3d] batch done: {start}~{start+len(chunk_names)-1} (chunk={len(chunk_names)}, ctx={use_ctx})")
            gc.collect()
            torch.cuda.empty_cache()

        print(f"end inpainting, result saved in {output_root_dir}")

        if super_resolution:
            print("start SR inpainting")
            del self.rgb_model
            gc.collect()
            torch.cuda.empty_cache()

            if not hasattr(self, "rgb_model_SR"):
                raise RuntimeError("[see3d] use_SR not enabled but super_resolution=True")

            height_sr = height_mvd * 2
            width_sr = width_mvd * 2

            # refs for SR
            ref_images_sr, _ = self.load_ref_images(source_imgs_dir, height_mvd, width_mvd)
            ref_images_sr = [im.convert("RGB").resize((width_sr, height_sr), resample=Image.BILINEAR) for im in ref_images_sr]
            gt_num_b_sr = len(ref_images_sr)
            full_mask_sr = Image.fromarray(np.ones((height_sr, width_sr), dtype=np.uint8) * 255).convert("L")
            ref_masks_sr = [full_mask_sr for _ in range(gt_num_b_sr)]

            prev_pred_sr = None

            for start in range(0, len(warp_names), batch_warp_num):
                chunk_names = warp_names[start:start + batch_warp_num]

                chunk_pred_imgs, chunk_masks = [], []
                for wn in chunk_names:
                    pred_path = os.path.join(output_root_dir, f"predict_{wn}")
                    if not os.path.exists(pred_path):
                        raise FileNotFoundError(f"[see3d][SR] Missing base predict: {pred_path}")

                    pred_img = Image.open(pred_path).convert("RGB").resize((width_sr, height_sr), resample=Image.BILINEAR)
                    mask_img = Image.open(os.path.join(warp_root_dir, self._mask_name_from_warp(wn))).convert("L").resize(
                        (width_sr, height_sr), resample=Image.NEAREST
                    )

                    chunk_pred_imgs.append(pred_img)
                    chunk_masks.append(mask_img)

                use_ctx = carry_over and (prev_pred_sr is not None)

                masks_sr = list(ref_masks_sr)
                warps_sr = list(ref_images_sr)

                if use_ctx:
                    masks_sr.append(full_mask_sr)
                    warps_sr.append(prev_pred_sr.convert("RGB").resize((width_sr, height_sr), resample=Image.BILINEAR))

                ctx_offset = 1 if use_ctx else 0
                gt_num_frames = gt_num_b_sr + ctx_offset

                masks_sr += chunk_masks
                warps_sr += chunk_pred_imgs

                prompt, batch = self.PIL2tensor(height_sr, width_sr, len(masks_sr), masks_sr, warps_sr, logicalNot=False)
                images_predict_sr_batch = self.rgb_model_SR.inference_next_frame(
                    prompt, batch,
                    len(masks_sr),
                    height_sr, width_sr,
                    gt_num_frames=gt_num_frames,
                    output_type='pil'
                )

                pred_chunk_sr = images_predict_sr_batch[gt_num_frames: gt_num_frames + len(chunk_names)]
                for img_pred, wn in zip(pred_chunk_sr, chunk_names):
                    img_pred.save(os.path.join(output_root_dir, f"SR_predict_{wn}"))

                prev_pred_sr = pred_chunk_sr[-1]
                print(f"[see3d][SR] batch done: {start}~{start+len(chunk_names)-1} (chunk={len(chunk_names)}, ctx={use_ctx})")

                gc.collect()
                torch.cuda.empty_cache()

            print(f"end SR inpainting, result saved in {output_root_dir}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--ref_imgs_dir', type=str, required=True)
    parser.add_argument('--warp_root_dir', type=str, required=True)
    parser.add_argument('--output_root_dir', type=str, required=True)
    parser.add_argument('--use_SR', action='store_true', help='Use super resolution for inpainting')

    parser.add_argument('--batch_warp_num', type=int, default=10, help='Number of warp frames to inpaint per batch')
    parser.add_argument('--no_carry_over', action='store_true', help='Disable ctx carry from previous batch')

    args = parser.parse_args()

    t1 = time.time()

    see3d = See3D(device='cuda', use_SR=args.use_SR)
    see3d.inpainting(
        source_imgs_dir=args.ref_imgs_dir,
        warp_root_dir=args.warp_root_dir,
        output_root_dir=args.output_root_dir,
        super_resolution=args.use_SR,
        batch_warp_num=args.batch_warp_num,
        carry_over=(not args.no_carry_over),
    )

    cat_save_root_path = os.path.join(os.path.dirname(args.output_root_dir), 'cat_img')
    os.makedirs(cat_save_root_path, exist_ok=True)

    warp_files = sorted(glob(os.path.join(args.warp_root_dir, "warp_frame*.png")))
    if len(warp_files) == 0:
        warp_files = sorted(glob(os.path.join(args.warp_root_dir, "warp_*.png")))
    warp_names = [os.path.basename(p) for p in warp_files]

    none_visible_rate_list = []

    for idx, wn in enumerate(warp_names):
        gs_render_img_path = os.path.join(args.warp_root_dir, wn)
        mask_img_path = os.path.join(args.warp_root_dir, wn.replace("warp_frame", "mask_frame").replace("warp_", "mask_"))
        inpaint_img_path = os.path.join(args.output_root_dir, f"predict_{wn}")

        if not (os.path.exists(gs_render_img_path) and os.path.exists(mask_img_path) and os.path.exists(inpaint_img_path)):
            print(f"[warn] missing for idx={idx}:")
            print(f"  {gs_render_img_path}")
            print(f"  {mask_img_path}")
            print(f"  {inpaint_img_path}")
            continue

        mask_img = np.array(Image.open(mask_img_path).convert("L")) / 255.0
        total_pixels = mask_img.shape[0] * mask_img.shape[1]
        mask_pixels = np.sum(mask_img)
        none_visible_rate = 1.0 - mask_pixels / max(total_pixels, 1)
        none_visible_rate_list.append(none_visible_rate)

        gs_render_img = Image.open(gs_render_img_path).convert("RGB")
        inpaint_img = Image.open(inpaint_img_path).convert("RGB")

        padding = 10
        cat_img = Image.new('RGB', (gs_render_img.width + inpaint_img.width + padding, gs_render_img.height))
        cat_img.paste(gs_render_img, (0, 0))
        cat_img.paste(inpaint_img, (gs_render_img.width + padding, 0))
        cat_img.save(os.path.join(cat_save_root_path, f'{idx:06d}-{none_visible_rate:.2f}.png'))

    if len(none_visible_rate_list) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(none_visible_rate_list, label='None Visible Rate')
        plt.xlabel('Frame Index')
        plt.ylabel('None Visible Rate')
        plt.title('None Visible Rate of GS Render and Inpaint')
        plt.legend()
        plt.savefig(os.path.join(cat_save_root_path, 'none_visible_rate.png'))
        plt.close()

    print(f'cat img saved in {cat_save_root_path}')
    t2 = time.time()
    print(f'Time cost: {t2 - t1:.2f}s')
