import types
from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from atria_core.logger.logger import get_logger
from atria_models.models.layoutlmv3.modeling_layoutlmv3 import (
    LayoutLMv3ForSequenceClassification,
)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import AutoTokenizer
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

# from atria.core.utilities.logging import get_logger
# from docxplainer.model_explainability_wrappers.sequence import (
#     SequenceModelExplainabilityWrapper,
# )
from atria_insights.explainer_pipelines.sequence_classification.base_pipeline import (
    SequenceClassificationExplanationPipeline,
)

logger = get_logger(__name__)

ATTN_MASKING_IDX = -1000


def get_special_tokens_mask(input_ids: torch.Tensor, tokenizer: AutoTokenizer):
    special_tokens_mask = torch.zeros_like(
        input_ids, dtype=torch.bool, device=input_ids.device
    )
    special_tokens_mask[input_ids == tokenizer.bos_token_id] = True
    special_tokens_mask[input_ids == tokenizer.eos_token_id] = True
    special_tokens_mask[input_ids == tokenizer.pad_token_id] = True
    return special_tokens_mask


def replace_embeddings(embeddings, baseline_embeddings, special_tokens_mask=None):
    batch_size, seq_len, _ = embeddings.size()

    # generate the mask probability matrix for masking
    replace_mask = torch.full((batch_size, seq_len), 1, device=embeddings.device)

    # set the mask probability of special tokens to be 0
    if special_tokens_mask is not None:
        replace_mask.masked_fill_(special_tokens_mask, value=0.0)

    # convert to bool
    replace_mask = replace_mask.bool().unsqueeze(-1).expand_as(embeddings)

    # expand the mask to the same size as the embeddings
    return embeddings * ~replace_mask + replace_mask * baseline_embeddings


class ExplainableLayoutLMv3TextEmbeddings:
    def forward(
        self,
        input_embeddings=None,
        position_embeddings=None,
        spatial_position_embeddings=None,
        token_type_embeddings=None,
    ):
        embeddings = input_embeddings + token_type_embeddings
        embeddings += position_embeddings
        embeddings = embeddings + spatial_position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ExplainableLayoutLMv3PatchEmbeddings:
    def forward(self, pixel_values, position_embedding=None):
        embeddings = self.proj(pixel_values)

        if position_embedding is not None:
            # interpolate the position embedding to the corresponding size
            position_embedding = position_embedding.view(
                1, self.patch_shape[0], self.patch_shape[1], -1
            )
            position_embedding = position_embedding.permute(0, 3, 1, 2)
            patch_height, patch_width = embeddings.shape[2], embeddings.shape[3]
            position_embedding = F.interpolate(
                position_embedding, size=(patch_height, patch_width), mode="bicubic"
            )
            embeddings = embeddings + position_embedding

        embeddings = embeddings.flatten(2).transpose(1, 2)
        return embeddings


class ExplainableLayoutLMv3:
    def forward_image(self, patch_embeddings):
        # add [CLS] token
        batch_size, seq_len, _ = patch_embeddings.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        patch_embeddings = torch.cat((cls_tokens, patch_embeddings), dim=1)

        # add position embeddings
        if self.pos_embed is not None:
            patch_embeddings = patch_embeddings + self.pos_embed

        patch_embeddings = self.pos_drop(patch_embeddings)
        patch_embeddings = self.norm(patch_embeddings)

        return patch_embeddings

    def forward(
        self,
        input_embeddings: Optional[torch.FloatTensor] = None,
        position_embeddings: Optional[torch.FloatTensor] = None,
        spatial_position_embeddings: Optional[torch.FloatTensor] = None,
        patch_embeddings: Optional[torch.FloatTensor] = None,
        token_type_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        input_shape = input_embeddings.size()[:-1]
        batch_size, seq_length = input_shape
        device = input_embeddings.device

        if input_embeddings is not None:
            if attention_mask is None:
                attention_mask = torch.ones(((batch_size, seq_length)), device=device)
            if bbox is None:
                bbox = torch.zeros(
                    tuple(list(input_shape) + [4]), dtype=torch.long, device=device
                )
            embedding_output = self.embeddings(
                input_embeddings=input_embeddings,
                position_embeddings=position_embeddings,
                spatial_position_embeddings=spatial_position_embeddings,
                token_type_embeddings=token_type_embeddings,
            )

        final_bbox = final_position_ids = None
        patch_height = patch_width = None
        if patch_embeddings is not None:
            visual_embeddings = self.forward_image(patch_embeddings)
            visual_attention_mask = torch.ones(
                (batch_size, visual_embeddings.shape[1]),
                dtype=torch.long,
                device=device,
            )
            # if the attention masking removal is enabled, remove the attention from the patch embeddings
            visual_attention_mask[:, 1:][
                patch_embeddings[:, :, 0] == ATTN_MASKING_IDX
            ] = 0

            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, visual_attention_mask], dim=1
                )
            else:
                attention_mask = visual_attention_mask

            if (
                self.config.has_relative_attention_bias
                or self.config.has_spatial_attention_bias
            ):
                if self.config.has_spatial_attention_bias:
                    visual_bbox = self._calc_visual_bbox(
                        device, dtype=torch.long, bsz=batch_size
                    )
                    if bbox is not None:
                        final_bbox = torch.cat([bbox, visual_bbox], dim=1)
                    else:
                        final_bbox = visual_bbox

                visual_position_ids = torch.arange(
                    0, visual_embeddings.shape[1], dtype=torch.long, device=device
                ).repeat(batch_size, 1)
                if input_embeddings is not None:
                    position_ids = torch.arange(
                        0, input_shape[1], device=device
                    ).unsqueeze(0)
                    position_ids = position_ids.expand(input_shape)
                    final_position_ids = torch.cat(
                        [position_ids, visual_position_ids], dim=1
                    )
                else:
                    final_position_ids = visual_position_ids

            if input_embeddings is not None:
                embedding_output = torch.cat(
                    [embedding_output, visual_embeddings], dim=1
                )
            else:
                embedding_output = visual_embeddings

            embedding_output = self.LayerNorm(embedding_output)
            embedding_output = self.dropout(embedding_output)
        elif (
            self.config.has_relative_attention_bias
            or self.config.has_spatial_attention_bias
        ):
            if self.config.has_spatial_attention_bias:
                final_bbox = bbox
            if self.config.has_relative_attention_bias:
                position_ids = self.embeddings.position_ids[:, : input_shape[1]]
                position_ids = position_ids.expand(input_shape)
                final_position_ids = position_ids

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, None, device, dtype=embedding_output.dtype
        )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        encoder_outputs = self.encoder(
            embedding_output,
            bbox=final_bbox,
            position_ids=final_position_ids,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            Hp=patch_height,
            Wp=patch_width,
        )

        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class ExplainableLayoutLMv3ForSequenceClassification:
    def forward(
        self,
        input_embeddings: Optional[torch.FloatTensor],
        position_embeddings: Optional[torch.FloatTensor],
        spatial_position_embeddings: Optional[torch.FloatTensor],
        patch_embeddings: Optional[torch.FloatTensor],
        token_type_embeddings: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor],
        bbox: Optional[torch.LongTensor],
        labels: Optional[torch.LongTensor],
    ):
        return_dict = True

        outputs = self.layoutlmv3(
            input_embeddings=input_embeddings,
            spatial_position_embeddings=spatial_position_embeddings,
            position_embeddings=position_embeddings,
            patch_embeddings=patch_embeddings,
            attention_mask=attention_mask,
            bbox=bbox,
            token_type_embeddings=token_type_embeddings,
            return_dict=return_dict,
        )

        sequence_output = outputs[0][:, 0, :]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ExplainableLayoutLMv3ForTokenClassification:
    def forward(
        self,
        input_embeddings: Optional[torch.FloatTensor],
        position_embeddings: Optional[torch.FloatTensor],
        spatial_position_embeddings: Optional[torch.FloatTensor],
        patch_embeddings: Optional[torch.FloatTensor],
        token_type_embeddings: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor],
        bbox: Optional[torch.LongTensor],
        labels: Optional[torch.LongTensor],
    ) -> Union[Tuple, TokenClassifierOutput]:
        outputs = self.layoutlmv3(
            input_embeddings=input_embeddings,
            spatial_position_embeddings=spatial_position_embeddings,
            position_embeddings=position_embeddings,
            patch_embeddings=patch_embeddings,
            attention_mask=attention_mask,
            bbox=bbox,
            token_type_embeddings=token_type_embeddings,
        )
        input_shape = input_embeddings.size()[:-1]

        seq_length = input_shape[1]
        # only take the text part of the output representations
        sequence_output = outputs[0][:, :seq_length]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LayoutLMv3SequenceClassificationExplanationPipeline(
    SequenceClassificationExplanationPipeline
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(
            self._model_pipeline.model,
            [
                LayoutLMv3ForSequenceClassification,
            ],
        ), (
            f"This explanation pipeline only supports LayoutLMv3ForSequenceClassification, but got {type(self._model_pipeline.model)}"
        )

    def __init__(
        self,
        group_tokens_to_words: bool = True,
        baselines_config: Optional[Dict[str, str]] = None,
        metric_baselines_config: Optional[Dict[str, str]] = None,
    ):
        super().__init__(model=model, group_tokens_to_words=group_tokens_to_words)
        assert isinstance(model, self._SUPPORTED_MODELS), (
            f"This explainability wrapper only supports the following models: {self._SUPPORTED_MODELS}"
        )
        self._patched_modules = []
        self._return_argmax_gathered_outputs = False
        self._baselines_config = baselines_config or {}
        self._metric_baselines_config = metric_baselines_config or {}

        self._tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlmv3-base")
        setattr(self.config, "mask_token_id", self._tokenizer.mask_token_id)

    def patch_forward_for_explainability(self):
        self._patched_modules = []

        def patch_func(module, other_module_class, func_name="forward"):
            original_method = getattr(module.__class__, func_name)
            patched_method = getattr(other_module_class, func_name)
            logger.debug(
                f"Patching {original_method} -> {patched_method}",
            )
            setattr(
                module,
                func_name,
                types.MethodType(patched_method, module),
            )
            self._patched_modules.append((module, original_method))

        # Patch forward methods
        if isinstance(self._model, LayoutLMv3ForSequenceClassification):
            patch_func(self._model, ExplainableLayoutLMv3ForSequenceClassification)
        elif isinstance(self._model, LayoutLMv3ForTokenClassification):
            patch_func(self._model, ExplainableLayoutLMv3ForTokenClassification)
            self._return_argmax_gathered_outputs = True

        # Patch other methods
        patch_func(self._model.layoutlmv3, ExplainableLayoutLMv3)
        patch_func(
            self._model.layoutlmv3, ExplainableLayoutLMv3, func_name="forward_image"
        )
        patch_func(
            self._model.layoutlmv3.embeddings,
            ExplainableLayoutLMv3TextEmbeddings,
        )
        patch_func(
            self._model.layoutlmv3.patch_embed, ExplainableLayoutLMv3PatchEmbeddings
        )

    def restore_forward(self):
        for module, original_method in self._patched_modules:
            logger.debug(
                f"Restoring {original_method}",
            )
            setattr(
                module,
                original_method.__name__,
                types.MethodType(original_method, module),
            )

    def prepare_explainable_inputs(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        bbox: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.LongTensor] = None,
        word_ids: Optional[torch.LongTensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if position_ids is None:
            # Create the position ids from the input token ids. Any padded tokens remain padded.
            position_ids = (
                self._model.layoutlmv3.embeddings.create_position_ids_from_input_ids(
                    input_ids, self._model.layoutlmv3.embeddings.padding_idx
                ).to(input_ids.device)
            )

        # prepare inputs
        return OrderedDict(
            input_embeddings=self._model.layoutlmv3.embeddings.word_embeddings(
                input_ids
            ),
            position_embeddings=self._model.layoutlmv3.embeddings.position_embeddings(
                position_ids
            ),
            spatial_position_embeddings=self._model.layoutlmv3.embeddings._calc_spatial_position_embeddings(
                bbox
            ),
            patch_embeddings=self._model.layoutlmv3.patch_embed(
                pixel_values
            ).contiguous(),
        )

    def _prepare_input_ids_baselines(self, input_ids, baselines_config):
        if "text" in baselines_config:
            inputs_embeds = self._model.layoutlmv3.embeddings.word_embeddings(input_ids)
            special_tokens_mask = get_special_tokens_mask(input_ids, self._tokenizer)
            baseline_type = baselines_config["text"]
            if baseline_type == "zero":
                baseline_embeddings = torch.zeros_like(inputs_embeds)
            elif baseline_type == "mask_token":
                baseline_embeddings = self._model.layoutlmv3.embeddings.word_embeddings(
                    torch.full_like(input_ids, self._tokenizer.mask_token_id)
                )
            elif baseline_type == "pad_token":
                baseline_embeddings = self._model.layoutlmv3.embeddings.word_embeddings(
                    torch.full_like(input_ids, self.padding_idx)
                )
            else:
                raise ValueError(
                    f"Invalid masking type: {baseline_type} for word embeddings. Supported types are 'zero' and 'pad'"
                )
            return replace_embeddings(
                embeddings=inputs_embeds,
                baseline_embeddings=baseline_embeddings,
                special_tokens_mask=special_tokens_mask,
            )

    def _prepare_position_ids_baselines(
        self, input_ids, position_ids, baselines_config
    ):
        if "position" in baselines_config:
            position_embeddings = self._model.layoutlmv3.embeddings.position_embeddings(
                position_ids
            )
            special_tokens_mask = get_special_tokens_mask(input_ids, self._tokenizer)
            baseline_type = baselines_config["position"]
            if baseline_type == "zero":
                baseline_embeddings = torch.zeros_like(position_embeddings)
            elif baseline_type == "pad_token":
                baseline_embeddings = (
                    self._model.layoutlmv3.embeddings.position_embeddings(
                        torch.full_like(input_ids, self._tokenizer.pad_token_id)
                    )
                )
            else:
                raise ValueError(
                    f"Invalid masking type: {baseline_type} for position embeddings. Supported types are 'zero' and 'pad'"
                )
            return replace_embeddings(
                embeddings=position_embeddings,
                baseline_embeddings=baseline_embeddings,
                special_tokens_mask=special_tokens_mask,
            )

    def _generate_spatial_position_baselines(self, input_ids, bbox, baselines_config):
        if "spatial_position" in baselines_config:
            spatial_position_embeddings = (
                self._model.layoutlmv3.embeddings._calc_spatial_position_embeddings(
                    bbox
                )
            )
            special_tokens_mask = get_special_tokens_mask(input_ids, self._tokenizer)
            baseline_type = baselines_config["spatial_position"]
            if baseline_type == "zero":
                baseline_embeddings = torch.zeros_like(spatial_position_embeddings)
            elif baseline_type == "pad_token":
                baseline_embeddings = self._calc_spatial_position_embeddings(
                    torch.zeros_like(bbox)
                )
            else:
                raise ValueError(
                    f"Invalid masking type: {baseline_type} for position embeddings. Supported types are 'zero' and 'pad'"
                )
            return replace_embeddings(
                spatial_position_embeddings,
                baseline_embeddings,
                special_tokens_mask,
            )

    def _generate_patch_embedding_baselines(self, image, baselines_config):
        if "image" in baselines_config:
            baseline_type = baselines_config["image"]
            image_embeddings = self._model.layoutlmv3.patch_embed(image)
            baseline_embeddings = None
            if (
                baseline_type == "mean_image"
            ):  # replace the image with zeros, zeros will correspond to the mean as the image is normalized
                baseline_embeddings = self._model.layoutlmv3.patch_embed(
                    torch.zeros_like(image)
                )
            elif (
                baseline_type == "max_image"
            ):  # replace the image with max value, which will correspond to the white color
                baseline_embeddings = self._model.layoutlmv3.patch_embed(
                    torch.ones_like(image) * image.max()
                )
            elif (
                baseline_type == "min_image"
            ):  # replace the image with max value, which will correspond to the white color
                baseline_embeddings = self._model.layoutlmv3.patch_embed(
                    torch.ones_like(image) * image.min()
                )
            elif baseline_type == "zero":
                baseline_embeddings = torch.zeros_like(image_embeddings)
            elif baseline_type == "attn_mask":
                baseline_embeddings = (
                    torch.ones_like(image_embeddings) * ATTN_MASKING_IDX
                )
            else:
                raise ValueError(
                    f"Invalid masking type: {baseline_type} for image embeddings. Supported types are 'zero', 'mask' and 'attn_mask'"
                )
            return baseline_embeddings

    def prepare_baselines_from_inputs(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        bbox: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.LongTensor] = None,
        word_ids: Optional[torch.LongTensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if position_ids is None:
            # Create the position ids from the input token ids. Any padded tokens remain padded.
            position_ids = (
                self._model.layoutlmv3.embeddings.create_position_ids_from_input_ids(
                    input_ids, self._model.layoutlmv3.embeddings.padding_idx
                ).to(input_ids.device)
            )

        return OrderedDict(
            input_embeddings=self._prepare_input_ids_baselines(
                input_ids, self._baselines_config
            ),
            position_embeddings=self._prepare_position_ids_baselines(
                input_ids, position_ids, self._baselines_config
            ),
            spatial_position_embeddings=self._generate_spatial_position_baselines(
                input_ids, bbox, self._baselines_config
            ),
            # notice here that the patch embeddings are generated from the pixel values first and then
            # the patch embeddings baselines are generated according to the patch embeddings shape
            patch_embeddings=self._generate_patch_embedding_baselines(
                pixel_values, self._baselines_config
            ),
        )

    def prepare_metric_baselines_from_inputs(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        bbox: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.LongTensor] = None,
        word_ids: Optional[torch.LongTensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if position_ids is None:
            # Create the position ids from the input token ids. Any padded tokens remain padded.
            position_ids = (
                self._model.layoutlmv3.embeddings.create_position_ids_from_input_ids(
                    input_ids, self._model.layoutlmv3.embeddings.padding_idx
                ).to(input_ids.device)
            )

        return OrderedDict(
            input_embeddings=self._prepare_input_ids_baselines(
                input_ids, self._metric_baselines_config
            ),
            position_embeddings=self._prepare_position_ids_baselines(
                input_ids, position_ids, self._metric_baselines_config
            ),
            spatial_position_embeddings=self._generate_spatial_position_baselines(
                input_ids, bbox, self._metric_baselines_config
            ),
            # notice here that the patch embeddings are generated from the pixel values first and then
            # the patch embeddings baselines are generated according to the patch embeddings shape
            patch_embeddings=self._generate_patch_embedding_baselines(
                pixel_values, self._metric_baselines_config
            ),
        )

    def prepare_feature_masks_from_inputs(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        bbox: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.LongTensor] = None,
        word_ids: Optional[torch.LongTensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # prepare feature masks
        token_level_feature_masks, special_token_ids_batch = (
            self._prepare_token_level_feature_masks(input_ids, word_ids)
        )
        bsz = pixel_values.shape[0]
        patch_size = 16
        total_patches = (pixel_values.shape[-1] // patch_size) ** 2
        patch_level_feature_masks = (
            torch.arange(
                total_patches,
                device=pixel_values.device,
            )
            .unsqueeze(0)
            .repeat(bsz, 1)
        )
        feature_masks = OrderedDict(
            input_embeddings=token_level_feature_masks.clone(),
            position_embeddings=token_level_feature_masks.clone(),
            spatial_position_embeddings=token_level_feature_masks.clone(),
            patch_embeddings=patch_level_feature_masks,
        )

        # accumulate feature mask indices over the features
        last_mask = None
        for key in feature_masks.keys():
            if last_mask is None:
                last_mask = feature_masks[key]
                continue
            feature_masks[key] = (
                feature_masks[key] + last_mask.max(dim=1, keepdim=True).values + 1
            )
            assert feature_masks[key].shape[0] == len(special_token_ids_batch)

            last_mask = feature_masks[key]

        frozen_features_per_type = {}
        for key in feature_masks.keys():
            if key == "patch_embeddings":
                continue
            for feature_mask, special_token_ids in zip(
                feature_masks[key], special_token_ids_batch
            ):
                if key not in frozen_features_per_type:
                    frozen_features_per_type[key] = []
                frozen_features_per_type[key].append(
                    feature_mask.min() + special_token_ids
                )
        frozen_features_per_type = list(frozen_features_per_type.values())
        frozen_features_batch = []
        for n in range(len(frozen_features_per_type[0])):
            frozen_features = []
            for i in range(len(frozen_features_per_type)):
                frozen_features.append(frozen_features_per_type[i][n])
            frozen_features = torch.cat(frozen_features)
            frozen_features_batch.append(frozen_features)

        total_features = max([mask.max().item() for mask in feature_masks.values()]) + 1
        return feature_masks, total_features, frozen_features_batch

    def expand_feature_masks_to_explainable_inputs(self, inputs, feature_masks):
        # expand feature masks to the same shape as the inputs
        for key, mask in feature_masks.items():
            feature_masks[key] = mask.unsqueeze(-1).expand_as(inputs[key])
        return feature_masks

    def prepare_additional_forward_kwargs(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        bbox: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.LongTensor] = None,
        word_ids: Optional[torch.LongTensor] = None,
    ):
        if token_type_ids is None:
            input_shape = input_ids.size()
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=input_ids.device
            )
        return dict(
            token_type_embeddings=self._model.layoutlmv3.embeddings.token_type_embeddings(
                self._prepare_token_type_ids_baselines(token_type_ids)
            ),
            attention_mask=attention_mask,
            bbox=bbox,
            labels=labels,
        )

    def forward(self, *args, **kwargs):
        model_outputs = self._model(*args, **kwargs)
        if self._is_output_explainable:
            logits = (
                model_outputs.logits
                if hasattr(model_outputs, "logits")
                else model_outputs
            )
            probs = self.softmax(logits)
            if len(probs.shape) == 3:
                probs = torch.gather(
                    probs,
                    2,
                    probs.argmax(dim=-1).unsqueeze(-1),
                ).squeeze(-1)
            return probs
        else:
            return model_outputs
