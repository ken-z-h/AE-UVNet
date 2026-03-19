from tensorflow.keras.layers import (
    Input, Conv3D, Conv3DTranspose, BatchNormalization,
    Activation, Dropout, Add, concatenate, Reshape,
    LayerNormalization, MultiHeadAttention, Dense,
    SpatialDropout3D, GlobalAveragePooling3D, multiply
)
from tensorflow.keras.models import Model


# ============================================================
# Common building blocks
# ============================================================

def se_block(input_tensor, reduction=8):
    """Squeeze-and-Excitation block for channel attention."""
    channels = input_tensor.shape[-1]
    se = GlobalAveragePooling3D()(input_tensor)
    se = Dense(max(channels // reduction, 1), activation='relu')(se)
    se = Dense(channels, activation='sigmoid')(se)
    se = Reshape((1, 1, 1, channels))(se)
    return multiply([input_tensor, se])


def residual_conv_block(x, filters, kernel_size=3, dropout=0.3):
    """Residual convolution block used in the U-Net style encoder."""
    conv = Conv3D(filters, kernel_size, padding='same')(x)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = SpatialDropout3D(0.05)(conv)

    conv = Conv3D(filters, kernel_size, padding='same')(conv)
    conv = BatchNormalization()(conv)
    if dropout > 0:
        conv = Dropout(dropout)(conv)

    shortcut = Conv3D(filters, kernel_size=1, padding='same')(x)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([shortcut, conv])
    x = Activation('relu')(x)
    return x


def vnet_conv_block(x, filters, kernel_size=5, dropout=0.0):
    """V-Net style convolution block with a larger kernel size."""
    conv = Conv3D(filters, kernel_size=kernel_size, padding='same')(x)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = SpatialDropout3D(0.05)(conv)

    if dropout > 0:
        conv = Dropout(dropout)(conv)

    return conv


def downsample_block(x, filters):
    """Downsampling block using strided convolution."""
    x = Conv3D(filters, kernel_size=2, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def upsample_block(x, filters):
    """Upsampling block using transposed convolution."""
    x = Conv3DTranspose(filters, kernel_size=2, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def transformer_block_bottleneck(x, num_heads=4, key_dim=16, reduce_channels=32):
    """Transformer bottleneck block applied at the deepest feature level."""
    x = Conv3D(reduce_channels, kernel_size=1, padding='same')(x)

    input_shape = x.shape
    spatial_dims = input_shape[1] * input_shape[2] * input_shape[3]

    x_reshaped = Reshape((spatial_dims, input_shape[-1]))(x)

    x_norm = LayerNormalization()(x_reshaped)
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x_norm, x_norm)
    x = Add()([x_reshaped, attn_output])

    x_norm2 = LayerNormalization()(x)
    mlp_output = Dense(units=input_shape[-1], activation='relu')(x_norm2)
    x = Add()([x, mlp_output])

    x = Reshape((input_shape[1], input_shape[2], input_shape[3], input_shape[-1]))(x)
    return x


# ============================================================
# Original model definitions
# The internal architectures are kept unchanged.
# ============================================================

def build_unet_transformer(input_shape=(128, 128, 128, 3), n_classes=4, base_filters=8):
    """Original U-Net with Transformer bottleneck."""
    inputs = Input(input_shape)

    e1 = residual_conv_block(inputs, base_filters)
    d1 = downsample_block(e1, base_filters * 2)

    e2 = residual_conv_block(d1, base_filters * 2)
    d2 = downsample_block(e2, base_filters * 4)

    e3 = residual_conv_block(d2, base_filters * 4)
    d3 = downsample_block(e3, base_filters * 8)

    e4 = residual_conv_block(d3, base_filters * 8)

    b = transformer_block_bottleneck(e4, num_heads=4, key_dim=16)
    b = Conv3D(base_filters * 8, kernel_size=3, padding='same')(b)
    b = BatchNormalization()(b)
    b = Activation('relu')(b)
    b = se_block(b)

    u1 = upsample_block(b, base_filters * 4)
    u1 = concatenate([u1, e3])
    u1 = Conv3D(base_filters * 4, kernel_size=3, padding='same')(u1)
    u1 = BatchNormalization()(u1)
    u1 = Activation('relu')(u1)
    u1 = se_block(u1)

    u2 = upsample_block(u1, base_filters * 2)
    u2 = concatenate([u2, e2])
    u2 = Conv3D(base_filters * 2, kernel_size=3, padding='same')(u2)
    u2 = BatchNormalization()(u2)
    u2 = Activation('relu')(u2)
    u2 = se_block(u2)

    u3 = upsample_block(u2, base_filters)
    u3 = concatenate([u3, e1])
    u3 = Conv3D(base_filters, kernel_size=3, padding='same')(u3)
    u3 = BatchNormalization()(u3)
    u3 = Activation('relu')(u3)
    u3 = se_block(u3)

    outputs = Conv3D(n_classes, kernel_size=1, activation='softmax')(u3)
    return Model(inputs, outputs, name='UNet_TransformerBottleneck')


def build_vnet_transformer(input_shape=(128, 128, 128, 3), n_classes=4, base_filters=8):
    """Original V-Net with Transformer bottleneck."""
    inputs = Input(input_shape)

    e1 = vnet_conv_block(inputs, base_filters)
    d1 = downsample_block(e1, base_filters * 2)

    e2 = vnet_conv_block(d1, base_filters * 2)
    d2 = downsample_block(e2, base_filters * 4)

    e3 = vnet_conv_block(d2, base_filters * 4)
    d3 = downsample_block(e3, base_filters * 8)

    e4 = vnet_conv_block(d3, base_filters * 8)

    b = transformer_block_bottleneck(e4, num_heads=4, key_dim=16)
    b = Conv3D(base_filters * 8, kernel_size=3, padding='same')(b)
    b = BatchNormalization()(b)
    b = Activation('relu')(b)
    b = se_block(b)

    u1 = upsample_block(b, base_filters * 4)
    u1 = concatenate([u1, e3])
    u1 = Conv3D(base_filters * 4, kernel_size=3, padding='same')(u1)
    u1 = BatchNormalization()(u1)
    u1 = Activation('relu')(u1)
    u1 = se_block(u1)

    u2 = upsample_block(u1, base_filters * 2)
    u2 = concatenate([u2, e2])
    u2 = Conv3D(base_filters * 2, kernel_size=3, padding='same')(u2)
    u2 = BatchNormalization()(u2)
    u2 = Activation('relu')(u2)
    u2 = se_block(u2)

    u3 = upsample_block(u2, base_filters)
    u3 = concatenate([u3, e1])
    u3 = Conv3D(base_filters, kernel_size=3, padding='same')(u3)
    u3 = BatchNormalization()(u3)
    u3 = Activation('relu')(u3)
    u3 = se_block(u3)

    outputs = Conv3D(n_classes, kernel_size=1, activation='softmax')(u3)
    return Model(inputs, outputs, name='VNet_TransformerBottleneck')


def build_uvnet(input_shape=(128, 128, 128, 3), n_classes=4, base_filters=8):
    """Original hybrid UV-Net architecture."""
    inputs = Input(input_shape)

    # U-Net branch
    ra1 = residual_conv_block(inputs, base_filters)
    dp1 = downsample_block(ra1, base_filters * 2)

    ra2 = residual_conv_block(dp1, base_filters * 2)
    dp2 = downsample_block(ra2, base_filters * 4)

    ra3 = residual_conv_block(dp2, base_filters * 4)
    dp3 = downsample_block(ra3, base_filters * 8)

    ra4 = residual_conv_block(dp3, base_filters * 8)

    # V-Net branch
    vb1 = vnet_conv_block(inputs, base_filters)
    dv1 = downsample_block(vb1, base_filters * 2)

    vb2 = vnet_conv_block(dv1, base_filters * 2)
    dv2 = downsample_block(vb2, base_filters * 4)

    vb3 = vnet_conv_block(dv2, base_filters * 4)
    dv3 = downsample_block(vb3, base_filters * 8)

    vb4 = vnet_conv_block(dv3, base_filters * 8)

    fused = concatenate([ra4, vb4])

    bottleneck = transformer_block_bottleneck(fused, num_heads=4, key_dim=16)
    bottleneck = Conv3D(base_filters * 8, kernel_size=3, padding='same')(bottleneck)
    bottleneck = BatchNormalization()(bottleneck)
    bottleneck = Activation('relu')(bottleneck)
    bottleneck = se_block(bottleneck)

    u1 = upsample_block(bottleneck, base_filters * 4)
    u1 = concatenate([u1, ra3, vb3])
    u1 = Conv3D(base_filters * 4, kernel_size=3, padding='same')(u1)
    u1 = BatchNormalization()(u1)
    u1 = Activation('relu')(u1)
    u1 = se_block(u1)

    u2 = upsample_block(u1, base_filters * 2)
    u2 = concatenate([u2, ra2, vb2])
    u2 = Conv3D(base_filters * 2, kernel_size=3, padding='same')(u2)
    u2 = BatchNormalization()(u2)
    u2 = Activation('relu')(u2)
    u2 = se_block(u2)

    u3 = upsample_block(u2, base_filters)
    u3 = concatenate([u3, ra1, vb1])
    u3 = Conv3D(base_filters, kernel_size=3, padding='same')(u3)
    u3 = BatchNormalization()(u3)
    u3 = Activation('relu')(u3)
    u3 = se_block(u3)

    outputs = Conv3D(n_classes, kernel_size=1, activation='softmax')(u3)
    return Model(inputs, outputs, name='Hybrid_DeepResUVNet')


def build_unet_plain(input_shape=(128, 128, 128, 3), n_classes=4, base_filters=8):
    """Original plain U-Net architecture without Transformer."""
    inputs = Input(input_shape)

    e1 = residual_conv_block(inputs, base_filters)
    d1 = downsample_block(e1, base_filters * 2)

    e2 = residual_conv_block(d1, base_filters * 2)
    d2 = downsample_block(e2, base_filters * 4)

    e3 = residual_conv_block(d2, base_filters * 4)
    d3 = downsample_block(e3, base_filters * 8)

    b = residual_conv_block(d3, base_filters * 8)

    u1 = upsample_block(b, base_filters * 4)
    u1 = concatenate([u1, e3])
    u1 = Conv3D(base_filters * 4, kernel_size=3, padding='same')(u1)
    u1 = BatchNormalization()(u1)
    u1 = Activation('relu')(u1)

    u2 = upsample_block(u1, base_filters * 2)
    u2 = concatenate([u2, e2])
    u2 = Conv3D(base_filters * 2, kernel_size=3, padding='same')(u2)
    u2 = BatchNormalization()(u2)
    u2 = Activation('relu')(u2)

    u3 = upsample_block(u2, base_filters)
    u3 = concatenate([u3, e1])
    u3 = Conv3D(base_filters, kernel_size=3, padding='same')(u3)
    u3 = BatchNormalization()(u3)
    u3 = Activation('relu')(u3)

    outputs = Conv3D(n_classes, kernel_size=1, activation='softmax')(u3)
    return Model(inputs, outputs, name='UNet_without_transformer')


def build_vnet_plain(input_shape=(128, 128, 128, 3), n_classes=4, base_filters=8):
    """Original plain V-Net architecture without Transformer bottleneck."""
    inputs = Input(input_shape)

    e1 = vnet_conv_block(inputs, base_filters, kernel_size=5, dropout=0.3)
    d1 = downsample_block(e1, base_filters * 2)

    e2 = vnet_conv_block(d1, base_filters * 2, kernel_size=5, dropout=0.3)
    d2 = downsample_block(e2, base_filters * 4)

    e3 = vnet_conv_block(d2, base_filters * 4, kernel_size=5, dropout=0.3)
    d3 = downsample_block(e3, base_filters * 8)

    b = vnet_conv_block(d3, base_filters * 8, kernel_size=5, dropout=0.3)

    u1 = upsample_block(b, base_filters * 4)
    u1 = concatenate([u1, e3])
    u1 = Conv3D(base_filters * 4, kernel_size=3, padding='same')(u1)
    u1 = BatchNormalization()(u1)
    u1 = Activation('relu')(u1)

    u2 = upsample_block(u1, base_filters * 2)
    u2 = concatenate([u2, e2])
    u2 = Conv3D(base_filters * 2, kernel_size=3, padding='same')(u2)
    u2 = BatchNormalization()(u2)
    u2 = Activation('relu')(u2)

    u3 = upsample_block(u2, base_filters)
    u3 = concatenate([u3, e1])
    u3 = Conv3D(base_filters, kernel_size=3, padding='same')(u3)
    u3 = BatchNormalization()(u3)
    u3 = Activation('relu')(u3)

    outputs = Conv3D(n_classes, kernel_size=1, activation='softmax')(u3)
    return Model(inputs, outputs, name='VNet_without_bottleneck')


# ============================================================
# Modular controller
# This layer only selects a model and does not modify any
# internal architecture.
# ============================================================

MODEL_REGISTRY = {
    'unet_plain': build_unet_plain,
    'vnet_plain': build_vnet_plain,
    'unet_transformer': build_unet_transformer,
    'vnet_transformer': build_vnet_transformer,
    'uvnet': build_uvnet,
}


def build_model(model_name='uvnet',
                input_shape=(128, 128, 128, 3),
                n_classes=4,
                base_filters=8):
    """
    Unified model selector without changing the original architectures.

    Supported models:
        - 'unet_plain'
        - 'vnet_plain'
        - 'unet_transformer'
        - 'vnet_transformer'
        - 'uvnet'
    """
    model_name = model_name.lower()

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model_name: {model_name}. "
            f"Available options: {list(MODEL_REGISTRY.keys())}"
        )

    builder = MODEL_REGISTRY[model_name]
    return builder(
        input_shape=input_shape,
        n_classes=n_classes,
        base_filters=base_filters
    )


# ============================================================
# Optional config-style interface
# This still preserves the original architectures by mapping
# valid configurations to existing model definitions only.
# ============================================================

def build_model_from_config(model_family='unet',
                            use_transformer=False,
                            use_hybrid=False,
                            input_shape=(128, 128, 128, 3),
                            n_classes=4,
                            base_filters=8):
    """
    Config-based interface that maps to the original models only.

    Rules:
        1. If use_hybrid=True, the model is UV-Net.
        2. If use_hybrid=False and model_family='unet':
           - use_transformer=False -> plain U-Net
           - use_transformer=True  -> U-Net + Transformer bottleneck
        3. If use_hybrid=False and model_family='vnet':
           - use_transformer=False -> plain V-Net
           - use_transformer=True  -> V-Net + Transformer bottleneck
    """
    model_family = model_family.lower()

    if use_hybrid:
        return build_uvnet(
            input_shape=input_shape,
            n_classes=n_classes,
            base_filters=base_filters
        )

    if model_family == 'unet':
        if use_transformer:
            return build_unet_transformer(
                input_shape=input_shape,
                n_classes=n_classes,
                base_filters=base_filters
            )
        return build_unet_plain(
            input_shape=input_shape,
            n_classes=n_classes,
            base_filters=base_filters
        )

    if model_family == 'vnet':
        if use_transformer:
            return build_vnet_transformer(
                input_shape=input_shape,
                n_classes=n_classes,
                base_filters=base_filters
            )
        return build_vnet_plain(
            input_shape=input_shape,
            n_classes=n_classes,
            base_filters=base_filters
        )

    raise ValueError("model_family must be 'unet' or 'vnet'")


# ============================================================
# Example
# ============================================================
if __name__ == '__main__':
    # Option 1: direct model name
    model = build_model(
        model_name='uvnet',
        input_shape=(128, 128, 128, 3),
        n_classes=4,
        base_filters=8
    )

    # Option 2: config-style control
    # model = build_model_from_config(
    #     model_family='unet',
    #     use_transformer=True,
    #     use_hybrid=False,
    #     input_shape=(128, 128, 128, 3),
    #     n_classes=4,
    #     base_filters=8
    # )

    model.summary()