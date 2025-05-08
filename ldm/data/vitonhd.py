import os
from os.path import join as opj

import cv2
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
import json
import torch
from PIL import Image
from transformers import CLIPImageProcessor
import random
def imread(
        p, h, w, 
        is_mask=False, 
        in_inverse_mask=False, 
        img=None
):
    if img is None:
        img = cv2.imread(p)
    if not is_mask:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w,h))
        img = (img.astype(np.float32) / 127.5) - 1.0  # [-1, 1]
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (w,h))
        img = (img >= 128).astype(np.float32)  # 0 or 1
        img = img[:,:,None]
        if in_inverse_mask:
            img = 1-img
    return img
def imread_for_albu(
        p, 
        is_mask=False, 
        in_inverse_mask=False, 
        cloth_mask_check=False, 
        use_resize=False, 
        height=512, 
        width=384,
):
    img = cv2.imread(p)
    if use_resize:
        img = cv2.resize(img, (width, height))
    if not is_mask:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = (img>=128).astype(np.float32)
        if cloth_mask_check:
            if img.sum() < 30720*4:
                img = np.ones_like(img).astype(np.float32)
        if in_inverse_mask:
            img = 1 - img
        img = np.uint8(img*255.0)
    return img
def norm_for_albu(img, is_mask=False):
    if not is_mask:
        img = (img.astype(np.float32)/127.5) - 1.0
    else:
        img = img.astype(np.float32) / 255.0
        img = img[:,:,None]
    return img

class VITONHDDataset(Dataset):
    def __init__(
            self, 
            data_root_dir, 
            img_H, 
            img_W, 
            is_paired=True, 
            is_test=False, 
            is_sorted=False, 
            transform_size=None, 
            transform_color=None,
            **kwargs
        ):
        self.drd = data_root_dir
        self.img_H = img_H
        self.img_W = img_W
        self.pair_key = "paired" if is_paired else "unpaired"
        self.data_type = "train" if not is_test else "test"
        self.is_test = is_test
        self.resize_ratio_H = 1.0
        self.resize_ratio_W = 1.0
        self.clip_processor = CLIPImageProcessor()

        self.resize_transform = A.Resize(img_H, img_W)
        self.transform_size = None
        self.transform_crop_person = None
        self.transform_crop_cloth = None
        self.transform_color = None

        #### spatial aug >>>>
        transform_crop_person_lst = []
        transform_crop_cloth_lst = []
        transform_size_lst = [A.Resize(int(img_H*self.resize_ratio_H), int(img_W*self.resize_ratio_W))]
    
        if transform_size is not None:
            if "hflip" in transform_size:
                transform_size_lst.append(A.HorizontalFlip(p=0.5))

            if "shiftscale" in transform_size:
                transform_crop_person_lst.append(A.ShiftScaleRotate(rotate_limit=0, shift_limit=0.2, scale_limit=(-0.2, 0.2), border_mode=cv2.BORDER_CONSTANT, p=0.5, value=0))
                transform_crop_cloth_lst.append(A.ShiftScaleRotate(rotate_limit=0, shift_limit=0.2, scale_limit=(-0.2, 0.2), border_mode=cv2.BORDER_CONSTANT, p=0.5, value=0))

        self.transform_crop_person = A.Compose(
                transform_crop_person_lst,
                additional_targets={"agn":"image", 
                                    "agn_mask":"image",
                                    "inpaint_mask":"image",
                                    "inpaint_image":"image",
                                    "GT_mask":"image",  
                                    "cloth_mask_warped":"image", 
                                    "cloth_warped":"image", 
                                    "image_densepose":"image", 
                                    "image_parse":"image", 
                                    "gt_cloth_warped_mask":"image",
                                    "pose":"image" 
                                    }
        )
        self.transform_crop_cloth = A.Compose(
                transform_crop_cloth_lst,
                additional_targets={"cloth_mask":"image"}
        )

        self.transform_size = A.Compose(
                transform_size_lst,
                additional_targets={"agn":"image", 
                                    "agn_mask":"image",
                                    "inpaint_mask":"image",
                                    "inpaint_image":"image",
                                    "GT_mask":"image", 
                                    "cloth":"image", 
                                    "cloth_mask":"image", 
                                    "cloth_mask_warped":"image", 
                                    "cloth_warped":"image", 
                                    "image_densepose":"image", 
                                    "image_parse":"image", 
                                    "gt_cloth_warped_mask":"image",
                                    "pose":"image"
                                    }
            )
        #### spatial aug <<<<

        #### non-spatial aug >>>>
        if transform_color is not None:
            transform_color_lst = []
            for t in transform_color:
                if t == "hsv":
                    transform_color_lst.append(A.HueSaturationValue(5,5,5,p=0.5))
                elif t == "bright_contrast":
                    transform_color_lst.append(A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.02), contrast_limit=(-0.3, 0.3), p=0.5))

            self.transform_color = A.Compose(
                transform_color_lst,
                additional_targets={"agn":"image",
                                    "inpaint_image":"image", 
                                    "cloth":"image",  
                                    "cloth_warped":"image",
                                    "pose":"image"
                                    }
            )
        #### non-spatial aug <<<<
                    
        assert not (self.data_type == "train" and self.pair_key == "unpaired"), f"train must use paired dataset"
        
        im_names = []
        c_names = []
        with open(opj(self.drd, f"{self.data_type}_pairs.txt"), "r") as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)
        if is_sorted:
            im_names, c_names = zip(*sorted(zip(im_names, c_names)))
        self.im_names = im_names
        
        self.c_names = dict()
        self.c_names["paired"] = im_names
        self.c_names["unpaired"] = c_names
        # ======= Get cloth caption of VITONHD dataset =======
        with open(
            os.path.join(data_root_dir, self.data_type, "vitonhd_" + self.data_type + "_tagged.json"), "r"
        ) as file1:
            data1 = json.load(file1)

        annotation_list = [
            # "colors",
            # "textures",
            "sleeveLength",
            "neckLine",
            "item",
        ]

        self.annotation_pair = {}
        for k, v in data1.items():
            for elem in v:
                annotation_str = ""
                for template in annotation_list:
                    for tag in elem["tag_info"]:
                        if (
                            tag["tag_name"] == template
                            and tag["tag_category"] is not None
                        ):
                            annotation_str += tag["tag_category"]
                            annotation_str += " "
                self.annotation_pair[elem["file_name"]] = annotation_str
        # ======================================================
    def __len__(self):
        return len(self.im_names)
    
    def __getitem__(self, idx):
        img_fn = self.im_names[idx]
        cloth_fn = self.c_names[self.pair_key][idx]
        if cloth_fn in self.annotation_pair:
            cloth_annotation = self.annotation_pair[cloth_fn]
        else:
            cloth_annotation = "shirts"
        if self.transform_size is None and self.transform_color is None:
            agn = imread(
                opj(self.drd, self.data_type, "agnostic-v3.2", self.im_names[idx]), 
                self.img_H, 
                self.img_W
            )
            agn_mask = imread(
                opj(self.drd, self.data_type, "agnostic-mask", self.im_names[idx].replace(".jpg", "_mask.png")), 
                self.img_H, 
                self.img_W, 
                is_mask=True, 
                in_inverse_mask=True
            )
            cloth = imread(
                opj(self.drd, self.data_type, "cloth", self.c_names[self.pair_key][idx]), 
                self.img_H, 
                self.img_W
            )
            cloth_mask = imread(
                opj(self.drd, self.data_type, "cloth-mask", self.c_names[self.pair_key][idx]), 
                self.img_H, 
                self.img_W, 
                is_mask=True, 
                cloth_mask_check=True
            )
            
            gt_cloth_warped_mask = imread(
                opj(self.drd, self.data_type, "gt_cloth_warped_mask", self.im_names[idx]), 
                self.img_H, 
                self.img_W, 
                is_mask=True
            ) if not self.is_test else np.zeros_like(agn_mask)

            image = imread(opj(self.drd, self.data_type, "image", self.im_names[idx]), self.img_H, self.img_W)
            image_densepose = imread(opj(self.drd, self.data_type, "image-densepose", self.im_names[idx]), self.img_H, self.img_W)
            pose = imread(opj(self.drd, self.data_type, "openpose_img", self.im_names[idx].replace('.jpg','_rendered.png')), self.img_H, self.img_W)

        else:
            agn = imread_for_albu(opj(self.drd, self.data_type, "agnostic-v3.2", self.im_names[idx]))
            agn_mask = imread_for_albu(opj(self.drd, self.data_type, "agnostic-mask", self.im_names[idx].replace(".jpg", "_mask.png")), is_mask=True)
            cloth = imread_for_albu(opj(self.drd, self.data_type, "cloth", self.c_names[self.pair_key][idx]))
            cloth_mask = imread_for_albu(opj(self.drd, self.data_type, "cloth-mask", self.c_names[self.pair_key][idx]), is_mask=True, cloth_mask_check=True)
            
            gt_cloth_warped_mask = imread_for_albu(
                opj(self.drd, self.data_type, "gt_cloth_warped_mask", self.im_names[idx]),
                is_mask=True
            ) if not self.is_test else np.zeros_like(agn_mask)
                
            image = imread_for_albu(opj(self.drd, self.data_type, "image", self.im_names[idx]))
            image_densepose = imread_for_albu(opj(self.drd, self.data_type, "image-densepose", self.im_names[idx]))
            pose = imread_for_albu(opj(self.drd, self.data_type, "openpose_img", self.im_names[idx].replace('.jpg','_rendered.png')))

            ## ======= Generate rectangle agnostic mask =======
            segment_map_path = opj(self.drd, self.data_type, "image-parse-v3", self.im_names[idx].replace(".jpg", ".png"))
            segment_map = Image.open(segment_map_path)
            segment_map = segment_map.resize((384,512), Image.NEAREST)
            parse_array = np.array(segment_map)
            garment_mask = (parse_array == 5).astype(np.float32) + \
                        (parse_array == 7).astype(np.float32)
            garment_mask_with_arms = (parse_array == 5).astype(np.float32) + \
                            (parse_array == 7).astype(np.float32) + \
                        (parse_array == 14).astype(np.float32) + \
                        (parse_array == 15).astype(np.float32)
            epsilon_randomness = random.uniform(0.001, 0.005)
            randomness_range = random.choice([ 80, 90, 100])
            kernel_size = random.choice([ 80, 100, 130, 150])


            # predict mask GT, inpainting mask to be dilated 
            garment_mask = 1 - garment_mask.astype(np.float32)
            garment_mask[garment_mask < 0.5] = 0
            garment_mask[garment_mask >= 0.5] = 1
            garment_mask_resized = cv2.resize(garment_mask, (384,512), interpolation=cv2.INTER_NEAREST)

        
            contours, _ = cv2.findContours(((1 - garment_mask_resized) * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if len(contours) != 0:
                max_contour = max(contours, key = cv2.contourArea)
                epsilon = epsilon_randomness * cv2.arcLength(max_contour, closed=True)  
                approx_contour = cv2.approxPolyDP(max_contour, epsilon, closed=True)
                randomness = np.random.randint(-randomness_range, randomness_range, approx_contour.shape)
                approx_contour = approx_contour + randomness

                zero_mask = np.zeros((512, 384))
                contours = [approx_contour]

                cv2.drawContours(zero_mask, contours, -1, (255), thickness=cv2.FILLED)

                kernel = np.ones((kernel_size,kernel_size),np.uint8)
                garment_mask_inpainting = cv2.morphologyEx(zero_mask, cv2.MORPH_CLOSE, kernel)
                garment_mask_inpainting = garment_mask_inpainting.astype(np.float32) / 255.0
                garment_mask_inpainting[garment_mask_inpainting < 0.5] = 0
                garment_mask_inpainting[garment_mask_inpainting >= 0.5] = 1
                garment_mask_inpainting = garment_mask_resized * (1 - garment_mask_inpainting)
            else:
                garment_mask_inpainting = np.zeros((512, 384))
            self.kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            garment_mask_GT = cv2.erode(garment_mask_resized, self.kernel_dilate, iterations=3)[None]
            garment_mask_inpainting = cv2.erode(garment_mask_inpainting, self.kernel_dilate, iterations=5)[None]

            garment_mask_GT_tensor = torch.from_numpy(garment_mask_GT)
            garment_mask_inpainting_tensor = torch.from_numpy(garment_mask_inpainting)


            # generate inpainting boundingbox, inpainting mask to be dilated, 
            garment_mask_with_arms = 1 - garment_mask_with_arms.astype(np.float32)
            garment_mask_with_arms[garment_mask_with_arms < 0.5] = 0
            garment_mask_with_arms[garment_mask_with_arms >= 0.5] = 1
            garment_mask_with_arms_resized = cv2.resize(garment_mask_with_arms, (384,512), interpolation=cv2.INTER_NEAREST)

            garment_mask_with_arms_boundingbox = cv2.erode(garment_mask_with_arms_resized, self.kernel_dilate, iterations=5)[None]


            # boundingbox
            _, y, x = np.where(garment_mask_with_arms_boundingbox == 0)
            if x.size > 0 and y.size > 0:
                x_min, x_max = np.min(x), np.max(x)
                y_min, y_max = np.min(y), np.max(y)
                boundingbox = np.ones_like(garment_mask_with_arms_boundingbox)
                boundingbox[:, y_min:y_max, x_min:x_max] = 0
            else:
                boundingbox = np.zeros_like(garment_mask_with_arms_boundingbox)

            boundingbox_tensor = torch.from_numpy(boundingbox)


            # limit in the boundingbox
            garment_mask_inpainting_tensor = torch.where((garment_mask_inpainting_tensor==0) & (boundingbox_tensor==0), torch.zeros_like(garment_mask_inpainting_tensor), torch.ones_like(garment_mask_inpainting_tensor))


            garment_mask = (parse_array == 5).astype(np.float32) + \
                            (parse_array == 7).astype(np.float32)

            garment_mask_with_arms = (parse_array == 5).astype(np.float32) + \
                            (parse_array == 7).astype(np.float32) + \
                        (parse_array == 14).astype(np.float32) + \
                        (parse_array == 15).astype(np.float32)
            if not self.is_test:
                mask_or_boundingbox = random.random()
                if mask_or_boundingbox < 1 - 0.4:
                    inpainting_mask_tensor = garment_mask_inpainting_tensor
                else:
                    inpainting_mask_tensor = boundingbox_tensor
            else:
                inpainting_mask_tensor = boundingbox_tensor
            ## ======= End generate rectangle agnostic mask =======
            if self.transform_size is not None:
                transformed = self.transform_size(
                    image=image, 
                    agn=agn, 
                    agn_mask=agn_mask,
                    # inpaint_image=inpaint_img,
                    inpaint_mask=inpainting_mask_tensor.permute(1,2,0).numpy().squeeze(-1)*255, 
                    GT_mask = garment_mask_GT_tensor.permute(1,2,0).numpy().squeeze(-1)*255,
                    cloth=cloth, 
                    cloth_mask=cloth_mask, 
                    image_densepose=image_densepose,
                    gt_cloth_warped_mask=gt_cloth_warped_mask,
                    pose=pose
                )
                image=transformed["image"]
                agn=transformed["agn"]
                agn_mask=transformed["agn_mask"]
                # inpaint_img =np.expand_dims(transformed["inpaint_mask"],axis = -1)*transformed["image"]
                inpaint_mask=transformed["inpaint_mask"]
                GT_mask = transformed["GT_mask"]
                image_densepose=transformed["image_densepose"]
                gt_cloth_warped_mask=transformed["gt_cloth_warped_mask"]
                pose=transformed["pose"]
                cloth=transformed["cloth"]
                cloth_mask=transformed["cloth_mask"]
                
            if self.transform_crop_person is not None:
                transformed_image = self.transform_crop_person(
                    image=image,
                    agn=agn,
                    agn_mask=agn_mask,
                    # inpaint_image=inpaint_img,
                    inpaint_mask=inpaint_mask,
                    GT_mask = GT_mask,
                    image_densepose=image_densepose,
                    gt_cloth_warped_mask=gt_cloth_warped_mask,
                    pose=pose,
                )

                image=transformed_image["image"]
                agn=transformed_image["agn"]
                agn_mask=transformed_image["agn_mask"]
                # inpaint_img =transformed_image["inpaint_image"]
                inpaint_mask=transformed_image["inpaint_mask"]
                GT_mask = transformed_image["GT_mask"]
                image_densepose=transformed_image["image_densepose"]
                gt_cloth_warped_mask=transformed["gt_cloth_warped_mask"]
                pose=transformed_image["pose"]

            if self.transform_crop_cloth is not None:
                transformed_cloth = self.transform_crop_cloth(
                    image=cloth,
                    cloth_mask=cloth_mask
                )

                cloth=transformed_cloth["image"]
                cloth_mask=transformed_cloth["cloth_mask"]

            agn_mask = 255 - agn_mask
            if self.transform_color is not None:
                transformed = self.transform_color(
                    image=image, 
                    agn=agn,
                    # inpaint_image=inpaint_img, 
                    cloth=cloth,
                    # pose=pose
                )

                image=transformed["image"]
                agn=transformed["agn"]
                cloth=transformed["cloth"]

                agn = agn * agn_mask[:,:,None].astype(np.float32)/255.0 + 128 * (1 - agn_mask[:,:,None].astype(np.float32)/255.0)
                
            agn = norm_for_albu(agn)
            agn_mask = norm_for_albu(agn_mask, is_mask=True)
            inpainting_mask = norm_for_albu(inpaint_mask, is_mask=True)
            image = norm_for_albu(image)
            inpaint_img = image * inpainting_mask
            GT_mask = norm_for_albu(GT_mask, is_mask=True)
            pose = norm_for_albu(pose)
            cloth = norm_for_albu(cloth)
            cloth_mask = norm_for_albu(cloth_mask, is_mask=True)
            image_densepose = norm_for_albu(image_densepose)
            gt_cloth_warped_mask = norm_for_albu(gt_cloth_warped_mask, is_mask=True)
            # ip_cloth = (cloth +1)/2.0*255
            # ip_cloth = np.clip(ip_cloth, 0, 255).astype(np.uint8)
            # cloth_pil = Image.fromarray(ip_cloth)

            cloth_pure = torch.from_numpy(cloth).permute(2,0,1)
            ref_list = [cloth_pure]
            image = torch.from_numpy(image).permute(2,0,1)
            GT_image_combined = torch.cat([image, cloth_pure], dim=-1)
            GT_mask = torch.from_numpy(GT_mask).permute(2,0,1)
            GT_mask_combined = torch.cat([GT_mask, torch.ones_like(GT_mask)], dim=-1)
            inpaint_img = torch.from_numpy(inpaint_img).permute(2,0,1)
            inpaint_img_combined = torch.cat([inpaint_img, cloth_pure], dim=-1)
            dense_pose = torch.from_numpy(image_densepose).permute(2,0,1)
            dense_pose_combined = torch.cat([dense_pose, cloth_pure], dim=-1)
            pose = torch.from_numpy(pose).permute(2,0,1) # check pose
            pose_combined = torch.cat([pose, cloth_pure], dim=-1)
            inpainting_mask_tensor = torch.from_numpy(inpainting_mask).permute(2,0,1)
            inpainting_mask_combined = torch.cat([inpainting_mask_tensor, torch.ones(inpainting_mask_tensor.shape, dtype=torch.float32)], dim=2)
        result = dict(
            image_name =img_fn,
            GT_image = GT_image_combined,
            GT_mask = GT_mask_combined,
            inpaint_image = inpaint_img_combined,
            inpaint_mask = inpainting_mask_combined,
            densepose = dense_pose_combined,
            ref_list = ref_list,
            posemap = pose_combined
        )
        return result