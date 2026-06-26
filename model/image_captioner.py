"""
This module is for the image captioner model. It comprises of two submodules,
    an image encoder (CNN) and caption decoder (transformer). The image captioner
    model takes in an image and predicts the caption for it.
"""
from typing import Optional

from function import \
    attention as attn, convolution as conv, \
    pool, regularization as reg, \
    update
from model import model as ml
from model.image_encoder import ImageEncoder
from model.caption_decoder import CaptionDecoder
from log import logger as log


IMAGE_ENCODER_DATA_FILENAME = 'image_captioner_image_encoder_data.pt'
CAPTION_DECODER_DATA_FILENAME = 'image_captioner_caption_decoder_data.pt'

class ImageCaptioner:
    def __init__(self,
        # Input files
        images_filename: Optional[str]=None,
        context_vectors_filename: Optional[str]=None,
        captions_filename: Optional[str]=None,

        # Base model hyperparameters
        batch_size=ml.BATCH_SIZE,
        num_folds=ml.NUM_FOLDS,
        num_epochs=ml.NUM_EPOCHS,
        learning_rate=update.LEARNING_RATE,
        reg_type=reg.REG_TYPE, reg_strength=reg.REG_STRENGTH,
        training_test_split=0.6,

        # CNN model hyperparameters
        num_encoder_in_channels=ml.NUM_IN_CHANNELS,
        num_encoder_out_features=ml.NUM_OUT_CLASSES,
        kernel_size=conv.KERNEL_SIZE,
        stride=conv.STRIDE, padding=conv.PADDING,
        pool_size=pool.KERNEL_SIZE,
        pool_stride=pool.STRIDE, pool_type=pool.POOL_TYPE,

        # Transformer model hyperparameters
        num_transformer_in_tokens=ml.NUM_IN_TOKENS,
        num_transformer_out_classes=ml.NUM_OUT_CLASSES,
        num_attn_heads=attn.NUM_ATTN_HEADS,
        dropout=reg.DROPOUT,

        # Model data files
        image_encoder_data_filename=IMAGE_ENCODER_DATA_FILENAME,
        load_image_encoder_data=False,
        caption_decoder_data_filename=CAPTION_DECODER_DATA_FILENAME,
        load_caption_decoder_data=False,

        # General parameters
        object_name: Optional[str]=None
    ):
        # Set the log id for the image encoder
        self.log_id = log.set_log_id(object_name, log.IMAGECAPTIONER)

        # If not loading the image encoder data, set the filename to None
        if not load_image_encoder_data:
            image_encoder_data_filename = None

        # If not loading the caption decoder data, set the filename to None
        if not load_caption_decoder_data:
            caption_decoder_data_filename = None

        # Store the base model hyperparameters
        base_model_hyperparameters = {
            'num_folds': num_folds,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'reg_type': reg_type,
            'reg_strength': reg_strength,
        }

        cnn_model_hyperparameters = {
            'num_in_channels': num_encoder_in_channels,
            'num_out_features': num_encoder_out_features,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'pool_size': pool_size,
            'pool_stride': pool_stride,
            'pool_type': pool_type,
        }

        transformer_model_hyperparameters = {
            'num_in_tokens': num_transformer_in_tokens,
            'num_out_classes': num_transformer_out_classes,
            'num_attn_heads': num_attn_heads,
            'dropout': dropout
        }

        # Set submodels for image encoding and caption decoding
        self.image_encoder = ImageEncoder(
            **cnn_model_hyperparameters,
            base_model_hyperparameters=base_model_hyperparameters,
            images_filename=images_filename,
            context_vectors_filename=context_vectors_filename,
            training_test_split=training_test_split,
            model_data_filename=image_encoder_data_filename,
            object_name='image_encoder'
        )

        self.caption_decoder = CaptionDecoder(
            **transformer_model_hyperparameters,
            base_model_hyperparameters=base_model_hyperparameters,
            captions_filename=captions_filename,
            training_test_split=training_test_split,
            model_data_filename=caption_decoder_data_filename,
            object_name='caption_decoder'
        )

        # Pretrain the image encoder
        self.image_encoder.train(
            use_patience=True,
            do_measure_accuracy=True,
            do_print_messages=True
        )

        """
        # Pretrain the caption decoder
        self.caption_decoder.train(
            use_patience=True,
            do_measure_accuracy=True,
            do_print_messages=True
        )
        """