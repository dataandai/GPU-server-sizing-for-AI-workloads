# LLM VRAM & KV Cache Monte Carlo Simulator

Engineering-focused simulator for estimating VRAM usage of large language models under different inference workloads, hardware choices, software stacks, and deployment profiles.

## Fő képességek

- **beégetett, kurált hardverkatalógus** 2 / 4 / 8 GPU-s konfigurációkkal
- **beégetett software stack katalógus**
- **deployment profile-ok** bare metal / VM / MIG jellegű futtatáshoz
- **dinamikus Hugging Face modellbetöltés**
- **Monte Carlo futtatás a web UI-ból egy gombbal**
- automatikus eredménymentés `output/results/` alá

## Projekt szerkezet

- `catalog/` – GPU, node, software stack és deployment seed adatok
- `scenarios/` – YAML scenariók
- `output/generated_scenarios/` – UI által generált scenariók
- `output/results/` – Monte Carlo eredményfájlok
- `ui.html` – webes felület
- `ui_server.py` – helyi HTTP szerver a UI futtatásához
- `run_all.py` – CLI futtató

## Gyors indulás Docker Compose-szal

### UI indítása

```bash
docker compose up --build ui
```

Ezután böngészőben:

```text
http://127.0.0.1:8765
```

Innen már működik a **Monte Carlo futtatása** gomb.

### Tesztek

```bash
docker compose --profile test up --build test
```

### CLI futtatás

```bash
docker compose --profile cli up --build cli
```

### Hardverkatalógus listázás

```bash
docker compose --profile catalog up --build catalog
```

## Docker nélkül

```bash
pip install -r requirements.txt
python ui_server.py
```

Böngésző:

```text
http://127.0.0.1:8765
```

## Hugging Face modellbetöltés

A scenario vagy a UI a következő mezőkkel tud dinamikus modellt kérni:

```yaml
model_source: "huggingface"
model_id: "Qwen/Qwen3-235B-A22B"
model_revision: "main"
model_local_files_only: false
```

A betöltő a következő fájlokat próbálja használni:

- `config.json`
- `generation_config.json`
- `README.md` / model card

### Private vagy gated modellek

Dockerben add át a tokenedet például így:

```bash
HF_TOKEN=hf_xxx docker compose up --build ui
```

vagy `.env` fájlban:

```env
HF_TOKEN=hf_xxx
```

A Compose ezt automatikusan átadja a konténernek.

### Offline cache mód

Ha csak lokális cache-ből akarsz dolgozni:

```yaml
model_local_files_only: true
```

A Compose a helyi Hugging Face cache-t is mountolja a konténerbe.

## Példa katalogus + HF scenario

Lásd: `examples/scenario_90_hf_catalog_example.yaml`

Részlet:

```yaml
scenario_name: "hf_qwen_on_dgx_b200"
model_source: "huggingface"
model_id: "Qwen/Qwen3-235B-A22B"
hardware_catalog_id: "nvidia_dgx_b200_8gpu"
software_stack_id: "nvidia_vllm_cuda"
deployment_profile_id: "bare_metal_container"
weight_precision: "int8"
kv_cache_precision: "fp8"
```

## Monte Carlo futtatás a UI-ból

A UI-ban:

1. válaszd ki a modellt, hardvert, stacket és deployment módot
2. generáld le a scenariót
3. nyomd meg a **Monte Carlo futtatása** gombot

A backend ilyenkor:
- elment egy YAML-t az `output/generated_scenarios/` alá
- lefuttatja a szimulációt
- elmenti a JSON-t az `output/results/` alá

A fájlnév egyszerűsített formátumú:

```text
modell_hardver_timestamp.json
```

például:

```text
qwen3-235b-a22b_nvidia-dgx-b200-8gpu_20260411_154746.json
```

## CLI használat

### Egy konkrét scenario futtatása

```bash
python run_all.py scenarios/scenario_01_fixed_batch.yaml
```

### JSON mentés automatikus névvel

```bash
python run_all.py scenarios/scenario_01_fixed_batch.yaml
```

Ez automatikusan ment `output/results/` alá.

### JSON mentés saját névvel

```bash
python run_all.py scenarios/scenario_01_fixed_batch.yaml --json-out output/results/custom.json
```

### Csak katalogus listázás

```bash
python run_all.py --list-hardware
python run_all.py --list-stacks
python run_all.py --list-deployments
```

## Fontos megjegyzések

- A hardver- és software-katalógus **statikus és kézzel kurált**.
- A Hugging Face modelloldal **dinamikus**.
- A jelenlegi modell elsődlegesen **memória, KV cache és OOM kockázat** becslő.
- A software stack és deployment profil most már a scenarióban és az eredményben is megjelenik, de a teljes throughput-/tokens-per-second modell még külön következő lépés.


## Gyors Docker hibaellenőrzés

Ha a böngészőben nem érhető el a felület:

```bash
docker compose up --build -d ui
docker compose ps
docker compose logs --tail=100 ui
```

Egészségellenőrzés a hostról:

```bash
curl http://localhost:${UI_PORT:-8765}/api/health
```

Vagy böngészőből:

```text
http://localhost:8765/api/health
```

Ha a `8765` port foglalt, indítsd másikon:

```bash
UI_PORT=8877 docker compose up --build -d ui
```

és akkor ezt nyisd meg:

```text
http://localhost:8877
```

Ha valami korábbi konténer beragadt:

```bash
docker compose down --remove-orphans
```


## Docker megjegyzes

A kontener indulaskor maga hozza letre az `output/` mappat, ezert annak nem kell elore leteznie a hoston.


## Új katalogus-struktúra

Az alkalmazás most már támogatja a rendezettebb katalogus-elrendezést is:

```text
catalog/
  hardware/
    gpu_catalog.json
    hardware_catalog.json
    software_stacks.json
    deployment_profiles.json
  blueprints/
    nvidia_blueprints_catalog.json
    nvidia_blueprints_catalog.schema.json
    nvidia_blueprint_templates.json
  models/
    hf_model_ingest_contract.json
  sources/
    source_manifest.md
```

A loader visszafelé kompatibilis a régi `catalog/*.json` fájlokkal is, de az új fejlesztések már az almappás szerkezetre épülnek.

## Új backend API-k a későbbi UI-hoz

A `ui_server.py` most már külön katalogus endpointokat is ad:

- `GET /api/catalog/summary`
- `GET /api/catalog/hardware`
- `GET /api/catalog/software-stacks`
- `GET /api/catalog/deployments`
- `GET /api/catalog/blueprints`
- `GET /api/catalog/blueprint-templates`
- `GET /api/catalog/blueprints/<blueprint_id>`

Ezek a következő UI-lépéshez készítik elő a blueprint alapú workload composer felületet.

## Új CLI lista opció

```bash
python run_all.py --list-blueprints
```


## Új workload-szimulációs motor

A projekt most már kétféle szcenáriót tud futtatni:
- a klasszikus VRAM/KV-cache scenariókat a `scenarios/` mappából,
- az általános workload-pipeline szimulációkat az `examples/workload_simulation/` mappából.

Példa futtatás:

```bash
python run_all.py examples/workload_simulation/general_retrieval_generation.yaml
```

A workload-szimulációs YAML-ek a `catalog/simulation/workload_simulation.schema.json` sémára épülnek,
és runtime/deployment-aware Monte Carlo motort használnak.

A UI szerver új séma-végpontja:
- `GET /api/catalog/workload-schema`
