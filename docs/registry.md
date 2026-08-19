# Registry

`registry/index.json` lists model-family artifact directories. All MP5 entries are explicit placeholder artifacts for detector, corners, quality, restoration, cleanup, and OCR families. They contain contract metadata only, no fake weights and no production claim.

```powershell
python -m wellfriend_models.registry.index registry/index.json --allow-placeholders
```

Production acceptance must validate a released artifact without the placeholder flag. The registry is an index, not a model download service.
