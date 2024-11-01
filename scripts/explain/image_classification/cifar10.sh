#!/bin/bash

python ./src/insightx/model_explainer.py \
    hydra.searchpath=[pkg://atria/conf,pkg://docsets/conf/pkg://torchxai/conf,./conf/] \
    output_dir=./output \
    data_module=huggingface \
    data_module.dataset_name=atria.data.datasets.cifar10.cifar10_huggingface_dataset.Cifar10HFDataset \
    data_module.dataset_output_key_map='{image: img, label: label}' \
    +data_transform@data_module.runtime_data_transforms.train=basic_image_aug \
    +data_transform@data_module.runtime_data_transforms.evaluation=basic_image_aug \
    +data_module.runtime_data_transforms.train.rescale_size=[224,224] \
    +data_module.runtime_data_transforms.evaluation.rescale_size=[224,224] \
    data_collator@data_module.train_dataloader_builder.collate_fn=batch_to_tensor \
    data_collator@data_module.evaluation_dataloader_builder.collate_fn=batch_to_tensor \
    task_module=image_classification_explanation_module \
    engine@test_engine=image_classification_test_engine \
    task_module.torch_model_builder.model_name=resnet50 test_engine.test_run=False \
    task_module.checkpoint=/home/aletheia/work/phd_projects/insightx/output/atria_trainer/Cifar10HFDataset/resnet50/2024-11-01/00-39-16/checkpoints/checkpoint_10.pt \
    data_module.max_test_samples=1000 \
    data_module.evaluation_dataloader_builder.batch_size=4 \
    $@ # task_module.checkpoint=/home/aletheia/work/phd_projects/insightx/output/atria_trainer/Cifar10HFDataset/resnet50/2024-10-31/23-58-14/checkpoints/checkpoint_10.pt \
