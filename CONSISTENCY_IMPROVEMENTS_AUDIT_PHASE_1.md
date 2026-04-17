# Konzisztenciajavítások — audit alapján, 1. fázis

Ez a kör nem új feature-ökre, hanem a döntési és kommunikációs réteg konzisztenciájára fókuszált.

## Beépített javítások

### 1. Közös backend nyelvi policy
Új modul: `src/language_policy.py`

Egységesítve lett a magyar nyelv miatti token/context szorzó backend oldalon.
Most már ugyanabból a policy-rétegből dolgozik:
- `blueprint_adapter.py`
- `template_adapter.py`
- `workload_simulation.py`
- `model_compatibility.py`

Ez csökkenti annak esélyét, hogy ugyanarra a magyar workloadra a compose, a kompatibilitási becslés és a futási eredmény eltérő token-terhelést számoljon.

### 2. Calibration coverage policy
Bővítve: `src/calibration_rules.py`

A `calibration_coverage_summary()` most már nem csak a coverage arányt adja vissza, hanem policy-szintű státuszt is:
- `uncalibrated`
- `partial_low`
- `partial_medium`
- `anchored`

Plusz mezők:
- `customer_label`
- `estimate_only`
- `severity`
- `summary_note`

Ez segít megakadályozni, hogy nulla vagy alacsony anchor-coverage mellett a report túl erős benchmark-jellegű állítást sugalljon.

### 3. Reporting truthfulness javítás
Bővítve:
- `src/reporting.py`
- `src/report_payload.py`
- `templates/report_customer_offer.html.j2`

Új elemek:
- top-level `calibration_summary` a report payloadban
- executive headline-ben calibration státusz
- külön kalibrációs blokk a HTML riportban
- detailed reportban coverage ratio + coverage status + note

### 4. Override-követés javítása
Bővítve:
- `src/blueprint_adapter.py`
- `src/template_adapter.py`
- `src/workload_simulation.py`
- `src/reporting.py`
- `src/report_payload.py`

A report most már nem heurisztikusan próbálja kitalálni, hogy történt-e modell override.
Közvetlenül átadott metaadatok:
- `requested_model_overrides`
- `override_role_ids`

Így a comparison nézetben az `override_active` már nem a model ID string formájából következtet.

### 5. Procurement decision consistency javítás
Bővítve: `src/procurement.py`

Korábban target arrival nélküli workload esetén is kijöhetett `FIT`, ha a safe capacity pozitív volt.
Ez félrevezető volt.
Most target nélküli esetben a fallback:
- `TUNE_OR_SCALE`

Ez jobban illeszkedik az audit-logikához és az üzleti értelmezéshez.

### 6. Flaky UI server tesztek stabilizálása
Új helper: `tests/http_test_utils.py`

A `time.sleep(0.1)` alapú szerverindítás-várás helyett socket-ready poll került be.
Ez stabilabb CI-futást ad lassabb környezetben is.

### 7. Új direkt unit tesztek
Új tesztfájl: `tests/test_procurement_and_reporting.py`

Lefedve:
- procurement decision band-ek
- safe/steady-state invariánsok
- procurement összefoglaló kiválasztási logika
- reporting `result_to_dict()` workload és memory branch
- report payload calibration truthfulness
- detailed report calibration mezők
- JSON export mezőkonzisztencia

## Eredmény

- teljes tesztcsomag: **80 passed**
- a legfontosabb konzisztenciajavítások most már nem csak dokumentációs szinten, hanem teszttel védve vannak

## Következő javasolt kör

1. `reporting.py` további célzott coverage növelése a family-specifikus detailed metric ágakra
2. compose contract tesztek blueprint + template útvonalra végigfuttatva
3. legacy `monte_carlo.py` / `kv_cache_model.py` státusz lezárása: tesztelt supported surface vagy hivatalos deprecated útvonal
4. catalog semantic validation suite
