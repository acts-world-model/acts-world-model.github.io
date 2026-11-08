"""vtm_cosmos: Cosmos3-Generator video–wrist–tactile–action world model.

Package layout (see cosmos3_visuotactile_world_model_implementation.md):
  streams   – stream registry, attention topology, time constants
  data      – latent/action cache adapter (read-only on existing trees)
  model     – token layout, stream/time mask, Cosmos3 Generator adapter, LoRA
  train     – noising, flow loss, training entry point
  infer     – sampler, VAE decode with anchor restore, rollout, metrics
"""

__version__ = "0.1.0"
