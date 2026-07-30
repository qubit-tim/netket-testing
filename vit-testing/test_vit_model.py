"""Shape-only smoke tests for the ViT building blocks (no physics checks)."""

import jax

from vit_model import Embed, Encoder, FMHA, ViT

# A batch of spin configurations on a 10x10 square lattice.
M = 200
L = 10
D_MODEL = 32
PATCH_SIZE = 2
N_HEADS = 8
NUM_LAYERS = 4


def _spin_configs(key):
    return jax.random.randint(key, shape=(M, L * L), minval=0, maxval=1) * 2 - 1


def test_embed_shape():
    key = jax.random.key(0)
    key, init_key, data_key = jax.random.split(key, 3)
    spin_configs = _spin_configs(data_key)

    embed_module = Embed(D_MODEL, PATCH_SIZE)
    params = embed_module.init(init_key, spin_configs)
    embedded_configs = embed_module.apply(params, spin_configs)

    n_patches = (L * L) // PATCH_SIZE**2
    assert embedded_configs.shape == (M, n_patches, D_MODEL)


def test_fmha_shape():
    key = jax.random.key(0)
    key, init_key, data_key = jax.random.split(key, 3)
    spin_configs = _spin_configs(data_key)

    embed_module = Embed(D_MODEL, PATCH_SIZE)
    embed_params = embed_module.init(init_key, spin_configs)
    embedded_configs = embed_module.apply(embed_params, spin_configs)
    n_patches = embedded_configs.shape[1]

    fmha_module = FMHA(D_MODEL, N_HEADS, n_patches)
    key, fmha_key = jax.random.split(key)
    fmha_params = fmha_module.init(fmha_key, embedded_configs)
    attention_vectors = fmha_module.apply(fmha_params, embedded_configs)

    assert attention_vectors.shape == embedded_configs.shape


def test_encoder_shape():
    key = jax.random.key(0)
    key, init_key, data_key = jax.random.split(key, 3)
    spin_configs = _spin_configs(data_key)

    embed_module = Embed(D_MODEL, PATCH_SIZE)
    embed_params = embed_module.init(init_key, spin_configs)
    embedded_configs = embed_module.apply(embed_params, spin_configs)
    n_patches = embedded_configs.shape[1]

    encoder_module = Encoder(NUM_LAYERS, D_MODEL, N_HEADS, n_patches)
    key, encoder_key = jax.random.split(key)
    encoder_params = encoder_module.init(encoder_key, embedded_configs)
    y = encoder_module.apply(encoder_params, embedded_configs)

    assert y.shape == embedded_configs.shape


def test_vit_shape():
    key, init_key, data_key = jax.random.split(jax.random.key(0), 3)
    spin_configs = _spin_configs(data_key)

    vit_module = ViT(NUM_LAYERS, D_MODEL, N_HEADS, PATCH_SIZE)
    params = vit_module.init(init_key, spin_configs)
    log_psi = vit_module.apply(params, spin_configs)

    assert log_psi.shape == (M,)
