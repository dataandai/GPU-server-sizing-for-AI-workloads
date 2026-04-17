# LLM VRAM & KV Cache Monte Carlo Simulator — Docker és futtatási útmutató

Ez a friss útmutató már az új UI szerveres működéshez készült. A `ui.html` önmagában csak statikus fájl; a **Monte Carlo futtatása** gombhoz a `ui_server.py` kell, ezért Dockerben is ezt érdemes indítani.

## 1. Belépés a projekt mappába

```bash
cd kvcachemontecarlo
```

## 2. UI indítása Docker Compose-szal

```bash
docker compose up --build ui
```

Böngészőben:

```text
http://127.0.0.1:8765
```

Itt már működik:
- YAML generálás
- Monte Carlo futtatás gombbal
- eredmény JSON automatikus mentése

## 3. Hol lesznek a fájlok?

A host gépeden:

- `output/generated_scenarios/` – UI által generált YAML-ek
- `output/results/` – Monte Carlo JSON eredmények

A result fájl neve ilyen lesz:

```text
modell_hardver_timestamp.json
```

például:

```text
qwen3-235b-a22b_nvidia-dgx-b200-8gpu_20260411_154746.json
```

## 4. Hugging Face token használata

### Ideiglenesen egy parancsra

```bash
HF_TOKEN=hf_xxx docker compose up --build ui
```

### .env fájlból

Hozz létre egy `.env` fájlt a projekt gyökerében:

```env
HF_TOKEN=hf_xxx
```

Majd:

```bash
docker compose up --build ui
```

## 5. Offline cache használata

Ha a scenario-ban ez van:

```yaml
model_local_files_only: true
```

akkor a program csak a helyi Hugging Face cache-ből dolgozik. A Compose ezt a cache-t mountolja ide:

```text
/root/.cache/huggingface
```

## 6. CLI futtatás Dockerből

### Egy scenario

```bash
docker compose run --rm cli python run_all.py scenarios/scenario_01_fixed_batch.yaml
```

### Saját HF + katalogus scenario

```bash
docker compose run --rm cli python run_all.py examples/scenario_90_hf_catalog_example.yaml
```

### Saját JSON kimeneti névvel

```bash
docker compose run --rm cli python run_all.py examples/scenario_90_hf_catalog_example.yaml --json-out output/results/sajat.json
```

### Hardver / stack / deployment listák

```bash
docker compose run --rm cli python run_all.py --list-hardware
docker compose run --rm cli python run_all.py --list-stacks
docker compose run --rm cli python run_all.py --list-deployments
```

## 7. Tesztek

```bash
docker compose --profile test up --build test
```

## 8. Ha csak sima docker run kell

### Image build

```bash
docker build -t llm-vram-sim .
```

### UI szerver indítása

```bash
docker run --rm -p 8765:8765 \
  -e UI_SERVER_HOST=0.0.0.0 \
  -e UI_SERVER_PORT=8765 \
  -e HF_TOKEN=${HF_TOKEN:-} \
  -v "$(pwd)/scenarios:/app/scenarios" \
  -v "$(pwd)/output:/app/output" \
  -v "$(pwd)/catalog:/app/catalog:ro" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  llm-vram-sim python ui_server.py
```

### CLI futtatás

```bash
docker run --rm \
  -e HF_TOKEN=${HF_TOKEN:-} \
  -v "$(pwd)/scenarios:/app/scenarios" \
  -v "$(pwd)/output:/app/output" \
  -v "$(pwd)/catalog:/app/catalog:ro" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  llm-vram-sim python run_all.py scenarios/scenario_01_fixed_batch.yaml
```

## 9. Gyakori hibák

### A böngésző nem éri el a UI-t

Nézd meg, hogy a konténer valóban fut-e, és a port publish megvan-e:

```bash
docker compose ps
```

A szervernek `0.0.0.0:8765`-ön kell figyelnie a konténerben.

### Hugging Face modell nem töltődik

Lehetséges okok:
- rossz `model_id`
- private/gated modellhez hiányzik a `HF_TOKEN`
- `model_local_files_only: true`, de a modell nincs a lokális cache-ben

### Nincs result JSON

A UI-s futtatás a `output/results/` alá ment. Ha CLI-t futtatsz, alapból szintén oda ír automatikus névvel, kivéve ha megadod a `--no-json-write` opciót.
