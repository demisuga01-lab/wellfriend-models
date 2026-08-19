# Registry

`registry/index.json` lists model-family artifact directories. MP6 retains explicit placeholder artifacts and adds low/mid/high no-weight **experimental** mobile entries for detector, corners, quality, restoration, and cleanup. They contain contract metadata only, no fake weights and no production claim.

```powershell
python -m wellfriend_models.registry.index registry/index.json --allow-placeholders --allow-nonproduction
```

Production acceptance must validate a released artifact without the placeholder flag. The registry is an index, not a model download service.
