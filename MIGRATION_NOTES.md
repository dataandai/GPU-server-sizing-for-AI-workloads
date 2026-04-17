# Migration notes

## Régi -> új útvonalak

- `catalog/gpu_catalog.json` -> `catalog/hardware/gpu_catalog.json`
- `catalog/hardware_catalog.json` -> `catalog/hardware/hardware_catalog.json`
- `catalog/software_stacks.json` -> `catalog/hardware/software_stacks.json`
- `catalog/deployment_profiles.json` -> `catalog/hardware/deployment_profiles.json`
- `catalog/hf_model_ingest_contract.json` -> `catalog/models/hf_model_ingest_contract.json`
- `catalog/source_manifest.md` -> `catalog/sources/source_manifest.md`

## Újonnan hozzáadott útvonalak

- `catalog/blueprints/nvidia_blueprints_catalog.schema.json`
- `catalog/blueprints/nvidia_blueprints_catalog.json`
- `catalog/blueprints/nvidia_blueprint_templates.json`
- `docs/blueprint_simulation_ux_notes.md`

## Megjegyzés

Ez a csomag csak az adatfájlok új helyét rendezi át.  
Ha szeretnéd, a következő lépésben ezt rá lehet drótozni a futó projektre is.
