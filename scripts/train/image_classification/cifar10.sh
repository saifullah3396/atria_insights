#!/bin/bash

atria.train \
    output_dir=./output/ \
    data_module=huggingface \
    data_module.dataset_name=atria.data.datasets.cifar10.cifar10_huggingface_dataset.Cifar10HFDataset \
    data_module.dataset_output_key_map='{image: img, label: label}' \
    +data_transform@data_module.runtime_data_transforms.train=basic_image_aug \
    +data_transform@data_module.runtime_data_transforms.evaluation=basic_image_aug \
    +data_module.runtime_data_transforms.train.rescale_size=[224,224] \
    +data_module.runtime_data_transforms.evaluation.rescale_size=[224,224] \
    data_collator@data_module.train_dataloader_builder.collate_fn=batch_to_tensor \
    data_collator@data_module.evaluation_dataloader_builder.collate_fn=batch_to_tensor \
    task_module.torch_model_builder.model_name=resnet50 test_engine.test_run=False \
    engine@validation_engine=image_classification_validation_engine \
    engine@test_engine=image_classification_test_engine \
    training_engine.max_epochs=10 \
    $@
