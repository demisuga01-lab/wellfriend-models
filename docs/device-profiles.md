# Device profiles

MP6 defines `low`, `mid`, `high`, `server`, `web`, and `unknown` device profiles. A profile is an engineering target, not a measured performance statement: allowed precision, model-size/memory/latency placeholders, resolution and tiling policy, runtime targets, and fallback strategy.

```powershell
python -c "from wellfriend_models.mobile import load_device_profile; print(load_device_profile('configs/device-profiles/low.json'))"
```

The profile names align with `wellfriend-perception` routing. Low/mid/high mobile artifacts use only their matching profiles; server and web remain future runtime targets.
