# 240 epochs ddim 
#paired
    # folder1="../inference_logs/VITONHD/VITONHD_release_input_person_combine_garment_240epochs_paired/2024-04-24T16-52-18/second_stage/result"
    # folder2="../datasets/VITONHD/test/image_512"
    # python ./PerceptualSimilarity/lpips_2dirs.py -d0 $folder1 -d1 $folder2 --use_gpu
    # python ./test_SSIM.py --folder1 $folder1 --folder2 $folder2

#unpaired
    folder1="/HDD/Projects/Try-on-Product/inference_logs/VITONHD_release_input_person_combine_garment_240epochs_paired/2025-05-05T10-41-29/second_stage/result"
    folder2="/workspace/Try-on-Product/datasets/zalando-hd-resized/test/image_512"
    python -m pytorch_fid $folder1 $folder2 --device cuda:0

# 0.07231199047248628   0.896280889575317   8.532824851216049

