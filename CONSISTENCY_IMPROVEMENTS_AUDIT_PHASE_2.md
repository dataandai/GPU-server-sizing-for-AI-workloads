# Consistency Improvements — Audit Phase 2

Ez a kör nem új funkciókra, hanem a támogatott felületek közötti szerződés-konzisztenciára fókuszált.

## Bekerült javítások

- **Compose → run → report contract tesztek**
  - blueprint flow és advanced template flow esetén is ellenőrizve van az override, a language policy és a report payload közötti konzisztencia.
- **Family-specifikus detailed metrics tesztek**
  - realtime voice és fraud blueprint workloadoknál külön ellenőrzött, hogy a detailed metrics valóban domain-specifikus üzemi mutatókat adnak.
- **Semantic catalog integrity validation**
  - új cross-reference validator ellenőrzi a hardware → GPU, hardware → stack/deployment, model profile → stack, blueprint → template → role és advanced template → role kapcsolatokat.
- **Legacy memory engine explicit surface policy**
  - a klasszikus VRAM/KV-cache Monte Carlo út most már explicit `deprecated_legacy_memory_engine` státuszt kap a result JSON-ban és a részletes riportban is.
- **Language scaling propagation a report felé**
  - a `language_scaling` metaadat most már a workload eredményen és a report payloadon is végigmegy, így a compare/report réteg nem veszti el a domináns nyelvi kontextust.

## Új / módosított fő fájlok

- `src/catalog_loader.py`
- `src/workload_simulation.py`
- `src/reporting.py`
- `src/report_payload.py`
- `src/monte_carlo.py`
- `run_all.py`
- `tests/test_consistency_contracts.py`

## Eredmény

- Teljes tesztfutás: **86 passed**
- A report és a compose pipeline közötti legfontosabb drift-kockázatok lezárva ezen a körön.

## Következő ajánlott kör

- `reporting.py` további family-ágainak célzott tesztelése
- `procurement.py` döntési határértékek parametrizált decision-table tesztjeinek bővítése
- `monte_carlo.py` és `kv_cache_model.py` mélyebb smoke/regression tesztek vagy hivatalos deprecation lezárás
