# Analisi preliminare: rango + cumulativa dei delta

## Risultato del 1 ottobre 2026

Analizzati 24.517 IGD 1x1, zero errori. Gli originali disponibili sono 50.000:
mancano 25.483 compressioni nella cartella analizzata. L'altra cartella
`data/cifar-10_compressed` non aggiunge nomi di immagini mancanti.
Rapporti dettagliati: `output/cumulative_audit_2026-10-01/` (esclusi da Git).

Numero di basi: minimo 2, mediana 932, percentile 95 circa 5.331,
percentile 99 circa 14.285, massimo 18.805 in `image_06405.igd`.

| S | Massimo z | Immagini oltre uint16 |
|---|---:|---:|
| 1 | 26.882 | 0 |
| 2 | 39.953 | 0 |
| 3 | 45.068 | 0 |
| 4 | 57.505 | 0 |
| 5 | 65.074 | 0 |
| 6 | 76.540 | 23 |

Massimo S intero comune per uint16: 5. Il massimo z a S=5 è in
`image_14525.igd`, diverso dall'immagine con più basi. Nessun S comune
per uint8: già a S=0, 18.280 immagini hanno un rango massimo oltre 255.
Ci sono 2.060 dizionari con gap tutti uguali; non sono automaticamente
i casi peggiori, perché conta anche il numero di basi.

Conclusione: è ragionevole provare S=1,2,3 in uint16. S=5 entra nei file
analizzati ma ha solo 461 livelli di margine; non è una garanzia su nuove
immagini o sui file mancanti. Il generatore futuro dovrà verificare gli overflow.
Non sono state generate nuove mappe né avviati addestramenti.

Questa analisi non crea la nuova rappresentazione e non modifica le compressioni.
Legge tutti gli IGD disponibili nella cartella specificata, un'immagine alla volta.
I risultati riguardano solo quei file: il confronto dei nomi con le immagini PNG
segnala la copertura, senza attestare che contenuto e parametri corrispondano.

## Esecuzione dalla radice del progetto

```powershell
cargo build --manifest-path src/igd_extract/Cargo.toml --bin analyze_dictionaries --target-dir src/gdcompress/target --release --offline --locked
python src/analyze_cumulative_dataset.py --output-dir output/cumulative_audit
```

La cartella di output deve essere nuova. Opzioni: `--input-dir`, `--original-dir`,
`--helper`. Default: `data/cifar-10_compressed_pg_1x1` e `data/cifar-10_resized`.
Il lettore Rust usa i codec originali; Python gestisce scansione e riepilogo.
Non servono librerie Python esterne. Non vengono creati JSON spaziali intermedi.

## Definizione

Le basi complete (costanti reinserite) sono ordinate con la stessa convenzione
dell'esperimento precedente: posizioni dei bit crescenti, primo carattere trattato
come MSB per definire il valore numerico. Non è un ordinamento per luminanza RGB.
Sono supportati gruppi 1x1 e basi fino a 32 bit; altri casi sono riportati come errori.

Per K basi ordinate, si considerano i K-1 gap positivi. Il delta iniziale è zero.

```
a[j] = log2(1 + delta[j]) / log2(1 + max_delta)
q[j] = round_half_up(S * a[j])
c_last = sum(q)
z_max = K - 1 + c_last
```

Normalizzazione in float64, somma in interi; nessuna pesatura per frequenza.
Il massimo S ammissibile è cercato con ricerca binaria, includendo S=0.
Per un dizionario con una sola base z è sempre zero: S non ha limite finito.
Gli esiti numerici usano la convenzione di arrotondamento effettiva del codice.
Non vengono verificati gli ID spaziali: questa è un'analisi dei dizionari,
non una verifica integrale del decoder.

## Output

- `per_image.jsonl`: K, bit/base, min/max gap, media dei gap log-normalizzati,
  indicatore di gap uguali, z_max per S=0..16, massimo S per uint8/uint16.
  La cumulativa finale si ricava come z_max - (K-1).
- `summary.json`: copertura, errori, distribuzione di K (quantili nearest-rank
  calcolati con indice arrotondato), immagini peggiori, overflow per S e massimo
  S comune. Il valore nullo va interpretato con i campi `unbounded` e
  `impossible_even_at_s0`.
- `missing_igd.txt`: nomi delle immagini originali senza compressione disponibile.

Il conteggio delle basi include tutte le righe del dizionario memorizzato.
I limiti sono validi sui file analizzati, non garantiti per immagini future.
Gli input e gli output di addestramento esistenti restano invariati.

## Verifica

```powershell
cargo test --manifest-path src/igd_extract/Cargo.toml --bin analyze_dictionaries --target-dir src/gdcompress/target --release --offline --locked
```

Test aritmetici: gap equidistanti, soglia di arrotondamento, impossibilità dovuta
al rango e dizionario a base unica. Controllo aggiuntivo eseguito su tre IGD in
`tmp/manual-test`: confronto con i dizionari JSON precedentemente esportati,
calcolo indipendente in Python di tutti i 17 massimi e verifica che S massimo
entri in uint16 mentre S+1 lo superi.
