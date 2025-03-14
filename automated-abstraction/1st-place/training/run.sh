#!/usr/bin/env sh

# set -euo pipefail

N_GPUS=1 # formerly 2
# all of these previous had this argument: --multi_gpu \

for fold in {0..4}; do
    # accelerate \
    #     launch \
    #         --num-processes "${N_GPUS}" \
    #         --mixed_precision fp16 \
    #             mlm.py \
    #                 --model-output-dir mlm \
    #                 --epochs 100 \
    #                 --batch-size 8 \
    #                 --lr 1e-5 \
    #                 --model-name google/bigbird-roberta-large \
    #                 --find-unused \
    #                 --fold $fold \
    #                 --block-size 512 \
    #                 --grad-check \
    #                 --data-path /competition-cdc-narratives/data/final/aa_track/public \


    # accelerate \
    #     launch \
    #         --num-processes "${N_GPUS}" \
    #         --mixed_precision fp16 \
    #             train.py \
    #                 --model-output-dir model_lovasz_mlm \
    #                 --model-name mlm/bigbird-roberta-large_f"${fold}"_e100_b8_ga1_oadamw_torch_fused_lr1_bs512 \
    #                 --local \
    #                 --epochs 400 \
    #                 --lr 1e-5 \
    #                 --batch-size 4 \
    #                 --grad-acc 1 \
    #                 --fold $fold \
    #                 --find-unused \
    #                 --grad-check \
    #                 --data-path /competition-cdc-narratives/data/final/aa_track/public \

    accelerate \
        launch \
            --num-processes "${N_GPUS}" \
            --mixed_precision fp16 \
                train.py \
                    --model-output-dir model_lovasz_mlm \
                    --model-name allenai/longformer-large-4096 \
                    --epochs 200 \
                    --lr 1e-5 \
                    --batch-size 2 \
                    --grad-acc 1 \
                    --fold $fold \
                    --data-path /competition-cdc-narratives/data/final/aa_track/public \

done
