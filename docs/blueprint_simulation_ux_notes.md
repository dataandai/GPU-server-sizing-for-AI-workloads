# Javasolt UX a blueprint-alapú workload szimulátorhoz

## 1. Template Composer
A felhasználó workload-template-et választ:
- NVIDIA blueprint alapú template
- saját / custom pipeline template

Az első képernyőn csak három döntést hozzon:
1. Mi a workload?
2. Mi a cél? (throughput / latency / 24x7 / N+1)
3. Melyik hardvercsaládot vizsgáljuk?

## 2. Model Binding
A template szerepköreihez modelleket rendel:
- detector / VLM / LLM / embedding / reranker / OCR / ASR
- alapértelmezett NVIDIA referencia modell látszik
- mellette "cseréld HF modellre" opció
- a "nyelvi megfelelés" külön jelző legyen (pl. hu-HU)

## 3. Capacity Inputs
A felhasználó nem alacsony szintű paramétereket ad meg először, hanem üzemi szintűeket:
- streamszám
- fps
- videóhossz / chunk
- QPS
- napi dokumentumszám
- párhuzamos agent futások
- SLA
- 24/7 / N+1

## 4. Results Explorer
Ne egyetlen JSON mezőlista legyen, hanem négy fő nézet:
- Feasibility
- Capacity
- Bottlenecks
- What-if

### Feasibility
- elfér / nem fér el
- kell-e másik node / több GPU / másik runtime

### Capacity
- max stream / QPS / párhuzamos futás
- p50 / p95 latency
- headroom

### Bottlenecks
- GPU compute
- VRAM
- decode
- retrieval
- agent orchestration
- network / storage

### What-if
- másik modell
- másik GPU
- másik batching policy
- MIG / VM / bare metal
- RAG on/off
- Hungarian-friendly model on/off

## 5. Beszerzési nézet
A végén kell egy procurement-fókuszú nézet:
- minimum működő konfiguráció
- recommended production configuration
- 24/7 safe configuration
- N+1 configuration
- risk score
