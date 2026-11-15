# Lernen in Spielen: Wann Hedge kreist, springt oder chaotisch wird – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-lernen-spiele-chaos-demo.streamlit.app/)** -->

Stück der Linie **Dynamische Systeme** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Es setzt die Stücke zu Periodenverdopplung (logistische Abbildung), Lyapunov-Exponent und Gleitkomma-Chaos voraus und ist die Gegenseite zu [noregret-demo](https://github.com/sebastian-hanisch/noregret-demo) (Kante `fix`): Deren README lässt Grenzzyklen und Chaos bei großer Schrittweite ausdrücklich offen (gelesen am 2026-11-14), die Demo zeigt Hedge nur dort, wo er konvergiert. Dieses Stück fährt die Schrittweite über die Schwelle und misst, was dann passiert. Verwandte Stücke der Linie: [logistische-abbildung-demo](https://github.com/sebastian-hanisch/logistische-abbildung-demo), [lyapunov-chaos-demo](https://github.com/sebastian-hanisch/lyapunov-chaos-demo), [gleitkomma-chaos-demo](https://github.com/sebastian-hanisch/gleitkomma-chaos-demo), [lager-chaos-demo](https://github.com/sebastian-hanisch/lager-chaos-demo).

## Warum dieses Problem

„Hedge ist in Überlastungsspielen chaotisch“ steht in der Literatur für eine Variante der multiplikativen Gewichte, und der Satz stimmt auf einer invarianten Diagonale, auf der beide Spieler exakt gleich starten und gleich lernen, in den gemessenen Beispielen aber nur in Beispiel 2 (in Beispiel 1 ist die Diagonalbahn ein stabiler 2-Zyklus) und nur für Teile des s-Bereichs (dazwischen liegen periodische Fenster). Die Kernfrage der Demo: Bleibt davon etwas übrig, wenn die Spieler nur *fast* gleich starten, und gibt es ein einfaches Spiel, in dem das Chaos für alle Anfangswerte bleibt?

**Kurzantwort (nur für die gemessenen Fälle, siehe Befunde):** Auf der symmetrischen Diagonale verliert Hedge den gemischten Gleichgewichtspunkt im Zwei-Spieler-Überlastungsspiel bei s = 8 (Beispiel 1) bzw. 17,78 (Beispiel 2) die Stabilität; quer zur Diagonale ist er bei s > 0 instabil (Querfaktor 1 + c > 1). Darüber gibt es Zyklen und auf der symmetrischen Diagonale in Beispiel 2 auch Chaos (in Beispiel 1 nicht). Die Diagonale ist aber quer instabil, und in den unsymmetrischen Messungen fand sich nach Nachprüfung keine chaotische Bahn: Chaos wurde in diesem Modell nur auf der symmetrischen Diagonale gemessen. Chaos für alle gemessenen Anfangswerte fand sich erst in einem anderen Verfahren (Hedge mit Vergessen) in einem 3×3-Spiel, in einer Spielfamilie und in einem schmalen Parameterfenster.

## Modell

- **Überlastungsspiel:** zwei Spieler, zwei Kanten, Kosten der Kante k gleich a_k · Last, Last gleich 1 plus 1, wenn der andere dieselbe Kante wählt. Beispiel 1: a₁ = a₂ = ½; Beispiel 2: a₁ = ¼, a₂ = 7/20.
- **Hedge** mit Schrittweite s: Gewichte w ← w · exp(−s · erwartete Kosten). Das ist MWU_ε mit 1 − ε = exp(−s), also s = −ln(1 − ε). Mit dem Logit u = ln(x/(1 − x)) gilt u ← u − s · (a₁(1 + x′) − a₂(2 − x′)), x′ ist die Wahrscheinlichkeit des Gegners für Kante 1.
- **Diagonale** u₁ = u₂: eine eindimensionale Abbildung. Gemischter Punkt x* = (2a₂ − a₁)/(a₁ + a₂); die Jacobi-Matrix hat dort die festen Eigenrichtungen (1, 1) und (1, −1) mit den Faktoren 1 − c (längs) und 1 + c (quer), c = s(a₁ + a₂)x(1 − x) > 0. Der gemischte Punkt verliert die Stabilität längs bei s = 2/((a₁ + a₂)x*(1 − x*)).
- **3×3-Spiele** R + e·I (R = Stein-Schere-Papier): Hedge im Dualraum, p ← (1 − g)p + s·A·y; g = 0 ist reines Hedge, g > 0 ein anderes Verfahren (Vergessen).
- Formeln und Symboltabelle stehen im Expander „📐 Mathematische Formulierung“ der App. Alle Zahlen gelten **in diesem Modell, bei diesen Parametern**.

## Methodik

- **Kern** (`lsc_algorithm.py`, `lsc_bimatrix.py`): überlaufsicher (tanh-Form der Sigmoidfunktion, Softmax mit abgezogenem Maximum), vektorisiert über Anfangswerte. Lyapunov-Exponent in der Diagonale als Mittel von ln|1 − c| und ln(1 + c) entlang der Bahn, im 2×2-Raum und in m×m-Spielen per Tangentenverfahren (Benettin) im Quotientenraum ohne die neutrale Richtung (1, …, 1), Gegenprobe Zwei-Bahn-Verfahren.
- **Klassifikation** einer Bahn: *reines Gleichgewicht* (ein Spieler x > 0,999, der andere x < 0,001), *still* (Spannweite im Messfenster unter 10⁻⁶, nicht rein), sonst *kreisend oder unregelmäßig*; *chaotisch* heißt in 3×3-Spielen: beschränkt, nicht konvergiert, λ > 0,003 **und** alle Wahrscheinlichkeiten > 10⁻⁹ (Inneres; Bahnen am Rand zählen nicht). Die Grenze 0,003 hängt an der Messdauer (Streuung des Exponenten); die Randgrenze 10⁻⁹ wurde nicht variiert.
- **Zufall:** nur über den eigenen SplitMix64 (`lsc_rng.py`, vektorisiert, gegen die Schleifenfassung und den bekannten Wert für Seed 0 getestet); Anfangswerte: Logits in [−3, 3]² (2×2), Dualkoordinaten in [−2, 2] (3×3). Zufalls-Seeds stehen in `lsc_messreihen.SEEDS`.
- **Vorgerechnet** (`tools/precompute_lsc.py` → `data/lsc_vorgerechnet.json`, einige Minuten): die Reihen der Befunde unten. In der App ist alles Vorgerechnete als „vorgerechnet“ gekennzeichnet; live rechnen die Experimente kleine Fassungen (je Eingabe unter etwa 2 Sekunden auf dem Rechner des Autors, Messung unten).
- **Schrittweiten** sind nie Zweierpotenzen und nie genau eine Schwelle: Presets und Tests nehmen 8,001, 7,99, 40,001 usw.; die vorgerechneten Reihen benutzen s + 0,00137. Bei s = 8 ist der Mehrfachwert −1 exakt, ein Gleitkomma-Artefakt, das hier vermieden wird.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stammen aus der Datei `data/lsc_vorgerechnet.json` (Reihe in Klammern) und werden in `tests/test_claims.py` nachgerechnet (Spalte „Test“; „Band“ heißt: der Test rechnet eine kleinere Fassung neu und prüft ein Band, die Zahl selbst steht nur in der Messung). Spiel ist, wenn nicht anders genannt, das Überlastungsspiel.

| Frage | Befund (Voraussetzung, Zählung) | Test |
|---|---|---|
| Schwelle des gemischten Punkts | geschlossen s = 8 (Beispiel 1) und 17,78 (Beispiel 2); erstes nicht konvergierendes s auf dem Raster mit Schritt 0,05 (je + 0,00137, Spannweite von x über 4 000 Schritte größer 10⁻⁵): 8,001 und 17,801 (`bifurkation`) | `test_claim_schwellen_geschlossen_und_gemessen` |
| Diagonale, Beispiel 1 | Periode 2 für alle gemessenen s von 8,0 bis 59,5 (104 Werte): der 2-Zyklus {u, −u} mit u = (s/4)·tanh(u/2) hat den Multiplikator (1 − (s/4)·sech²(u/2))², der für s > 8 (bis 60 nachgerechnet) unter 1 liegt und sich mit wachsendem s der 1 nähert (ab etwa s = 44 konvergiert die Bahn zu langsam für 20 000 Einschwingschritte, die Periodenerkennung erkennt den Zyklus dann an der monotonen Annäherung, siehe Korrektur 7); keine Periodenverdopplung; **0 von 104** Werten oberhalb der Schwelle chaotisch (λ > 0,003, 60 000 Schritte, Raster 1 bis 59,5 mit Schritt 0,5, je + 0,00137) (`bifurkation`) | `test_claim_diagonale_beispiel_1_kein_chaos` |
| Diagonale, Beispiel 2 | **34 von 84** Werten oberhalb der Schwelle chaotisch (gleiches Raster), erster bei s = 26,0; periodische Fenster der Perioden 2 (18,0 bis 23,0; 11 Werte), 4 (23,5 bis 59,5; 28 Werte), 3 (31,0 bis 32,5; 4 Werte), außerdem je ein Wert oder zwei mit den Perioden 5, 6, 10, 12, 28 und 48 (`bifurkation`) | `test_claim_diagonale_beispiel_2_chaos_und_fenster` |
| Diagonale quer instabil | Beispiel 2, s = 20 / 26 / 30 / 40 / 50 (je + 0,00137): λ_quer = +1,0392 / +1,0191 / +1,0365 / +0,8311 / +0,4005, λ_längs = −1,1208 / +0,1787 / +0,3940 / +0,3680 / −0,7421 (60 000 Schritte nach 4 000 Einschwingschritten); der Querfaktor 1 + c ist wegen c > 0 in jedem Schritt größer als 1 (`quer`) | `test_claim_querexponent_und_wachstum` |
| Wachstum eines Anfangsunterschieds | Beispiel 2, s = 40,001: Abstand 1 nach 9 / 18 / 26 / 35 Schritten für δ = 10⁻³ / 10⁻⁶ / 10⁻⁹ / 10⁻¹², gemessene Rate +0,63 / +0,76 / +0,81 / +0,80 je Schritt (Einschwingen 4 000 Schritte). Ohne Einschwingen (Start bei u = 0,37): 8 / 14 / 21 / 30 Schritte, Rate +0,95 für 10⁻¹² (Übergang, nicht die Rate der Bahn) (`quer`) | `test_claim_querexponent_und_wachstum` (Band) |
| Unsymmetrische Starts, zwei Beispiele | **2 400 von 2 400** Starts (6 Konfigurationen × 400) enden in einem reinen Gleichgewicht: Beispiel 1 bei s = 5, 10, 40; Beispiel 2 bei s = 10, 20, 40 (`asym_b`) | `test_claim_unsymmetrisch_reines_gleichgewicht` (Band: 6 × 100) |
| Asymmetrie von Schrittweite oder Kosten | Beispiel 2, s₁ = 26 und 40; Spieler 2 mit Schrittweite s₁(1 + δ) oder Kosten a₂(1 + δ) auf Kante 2, δ = 10⁻⁹ bis 0,1: **20 Konfigurationen × 2 000 Starts = 40 000 von 40 000** zufälliger Starts (Logits in [−3, 3]²) in reinem Gleichgewicht; auch ohne Abweichung (δ = 0) enden 1 900 von 1 900 zufällig gestarteten Bahnen rein (die 100 Starts mit u₂ = u₁ bleiben auf der Diagonale): die Abweichung δ ist dafür nicht nötig, sie ändert nur, was die exakt symmetrisch gestartete Bahn tut (`asym_c`) | `test_claim_asymmetrie_20_konfigurationen` (Band: 20 × 300) |
| Diagonale bleibt in Gleitkomma exakt | 100 Starts mit u₂ = u₁ je Variante und s₁: alle 100 bleiben nach 5 500 Schritten auf der Diagonale (bitgleich); ein Unterschied von 10⁻¹² verlässt sie nach 35 Schritten (`asym_c`, `quer`) | `test_claim_diagonale_bleibt_in_gleitkomma_exakt` |
| Gegenprobe zufällige Spiele | 3 × 20 000 Spiele mit zufälligen Kosten a₁, a₂, b₁, b₂ ∈ (0,05; 1), Schrittweiten ∈ (2; 80), unsymmetrischem Start (Abstand der Logits u₁, u₂ mindestens 0,01), 6 000 + 2 000 Schritten: reines Gleichgewicht 17 974 / 10 981 / 10 982, still (nicht rein) 2 026 / 9 014 / 9 015, weder noch 0 / 5 / 3; davon λ > 0,003: 0 / 1 / 1, und diese beiden sind bei 30 000 Einschwingschritten kein Chaos mehr (`v9g`) | `test_claim_gegenprobe_zufaellige_spiele` (Band: 3 × 2 000 Spiele) |
| Nullsumme (Stein-Schere-Papier) | mittleres ln det je Schritt im Dualraum für ε = 0,01 / 0,05 / 0,1 / 0,3 / 1,0: +4,3·10⁻⁵ / +1,5·10⁻⁶ / +4,0·10⁻¹⁰ / 0 / 0; KL-Abstand zum Gleichgewicht am Ende 1,59 / 26,3 / 55,3 / 171 / 575 (10⁵ Schritte, ein Start); größter Exponent im Quotientenraum +9,2·10⁻⁵ / +2,4·10⁻⁴ / +1,9·10⁻⁴ für s = 0,01 / 0,1 / 1,0 (20 000 Schritte), also positiv, aber unter der Chaos-Grenze 0,003; kleinste Wahrscheinlichkeit am Ende bei ε = 0,01 gleich 0,035 (die Bahn liegt im Inneren), bei ε ≥ 0,05 unter 10⁻¹³ (am Rand) (`nullsumme`) | `test_claim_nullsumme` (Band, 20 000 Schritte) |
| 3×3, reines Hedge | Spiele R + e₁·I und R + e₂·I (25 Spiele, e₁, e₂ ∈ {−0,4 … 0,4}) × 40 Starts × 4 Schrittweiten (0,3 / 1 / 3 / 10): **0 von 4 × 1 000** Bahnen chaotisch im Inneren; zufällige 3×3-Spiele (Korrelation γ = −0,8 / −0,4 / 0 / +0,4, je 300 Spiele × 8 Starts, s = 0,5 / 2 / 8): **0 von 12 × 2 400** (`hedge_rpe`, `hedge_random`); die Zählung „konvergiert“ hängt an der Messdauer: bei dreifacher Dauer (12 000 + 12 000 statt 4 000 + 4 000 Schritte) steigt sie in `hedge_rpe` von 140 / 172 / 194 / 179 auf 412 / 411 / 454 / 475 von je 1 000, „0 chaotisch“ bleibt (unabhängige Nachrechnung, nicht in der Messreihe) | `test_claim_hedge_3x3_kein_chaos` (Band: 4 × 200 Starts) |
| Bahnen am Rand, 3×3 reines Hedge | ohne die Randgrenze 10⁻⁹ zählt dieselbe Messung 67 / 86 / 170 / 253 von je 1 000 Bahnen mit λ > 0,003 (s = 0,3 / 1 / 3 / 10); das sind Bahnen am Rand, die **nicht** als Chaos gezählt und nicht gedeutet werden; bei dreifacher Messdauer bleiben von ihnen nur 2 / 0 / 9 / 18 über der Grenze (höchstens 7 %; unabhängige Nachrechnung, nicht in der Messreihe) (`hedge_rpe`, Feld `loose`) | `test_claim_hedge_3x3_kein_chaos` |
| Mit Vergessen: inneres Chaos | Spiel e₁ = 0, e₂ = +0,4, g = 0,02, s = 0,50137 (0,5 + 0,00137): **400 von 400** zufälligen Starts mit λ > 0,003 bei Messdauer 2 000, 6 000 und 18 000; λ-Quantile 0 / 0,5 / 1 bei 18 000: +0,0319 / +0,0337 / +0,0359; Zwei-Bahn- und Tangentenverfahren stimmen auf 0,0007 je Start überein (100 Starts, Median +0,0336 beide); kleinste Wahrscheinlichkeit entlang der Bahn log₁₀ −5,03 bis −4,58, Spannweite von x im Median 0,987 (`vergessen`) | `test_claim_vergessen_haupt` (Band: 100 Starts) |
| Mit Vergessen: Fenster | Anteil chaotischer Starts (je 100) bei g = 0,005 / 0,01 / 0,02 / 0,03 / 0,05 / 0,08: 0 / 0,31 / 1,00 / 0,98 / 0 / 0 (bei 0,08 alles konvergent); bei s = 0,1 / 0,2 / 0,35 / 0,5 / 0,7 / 1,0 / 1,5: 0 / 0 / 0 / 1,00 / 0,32 / 0 / 0 (bei 0,1 alles konvergent); Spielgitter e₁ −0,2 bis 0,2 × e₂ 0 bis 0,6 (je 40 Starts): **17 von 35** Zellen mit Anteil ≥ 0,95, untere Grenze bei e₁ + e₂ = 0,3 (kleinere Summen: 0), obere Grenze unregelmäßig; in den anderen fünf Konfigurationen (g = 0,02 mit s = 2 und 8; g = 0,1 mit s = 0,5, 2, 8; je 25 Spiele × 40 Starts): 0 von 5 × 1 000; die Konfiguration g = 0,02, s = 0,5: 203 von 1 000 in 6 von 25 Spielen (`vergessen`) | `test_claim_vergessen_fenster` (Band) |
| Rechenzeit der Live-Teile | jede Live-Rechnung der App bleibt an den größten Reglerwerten unter 10 Sekunden (Obergrenze im Test; die gemessene Zeit ist plattformabhängig und steht nicht in dieser README) | `test_claim_rechenzeit_am_reglermaximum` (Timeout-Band) |

## Befunde und Korrekturen gegenüber dem Plan

Der Plan (`PLAN-09`, Vormessung mit dem Zufallsgenerator von numpy) wurde im Bau mit dem SplitMix64 dieser Demo neu gemessen; die Messlogik ist neu geschrieben. Die neue Messung gilt:

1. **Wachstumsrate und Schritte bis zum Verlassen der Diagonale.** Die Vormessung nannte „30 Schritte bei 10⁻¹²“ (Start sofort bei u = 0,37) und getrennt die Rate +0,816 (nach 4 000 Einschwingschritten). Beides gehört zu verschiedenen Messungen: Reproduziert wurde 30 Schritte ohne Einschwingen, dort ist die Rate +0,95 (Übergangsphase). Nach 4 000 Einschwingschritten, also auf der Bahn, sind es 35 Schritte bei Rate +0,80, nahe am Querexponent +0,83. Die App und die Verdikte benutzen die Fassung nach dem Einschwingen.
2. **Gegenprobe zufälliger Spiele: kein „keine einzige“.** Die Vormessung zählte 0 / 6 / 2 Bahnen „weder still noch rein“ und keine mit λ > 0,003. Im Bau sind es 0 / 5 / 3, davon 0 / 1 / 1 mit λ > 0,003 nach 6 000 + 2 000 Schritten. Beide Bahnen sind langsame Übergänge zu einem reinen Gleichgewicht (bei 30 000 Einschwingschritten rein, λ ≈ 0). Der Satz bleibt: keine chaotische Bahn gemessen; er gilt für die Messdauer und die Klasse zufälliger 2×2-Überlastungsspiele.
3. **Spielgitter des Vergessens:** 17 von 35 Zellen mit Anteil ≥ 0,95 (die Vormessung nannte 17 von 35 bei ≥ 0,97; bei ≥ 0,97 sind es hier 15); die untere Grenze e₁ + e₂ = 0,3 bleibt, die obere ist unregelmäßig (Zelle e₁ = 0,1, e₂ = 0,5: 0,125 hier, 0,05 in der Vormessung). Die Anteile im g-Fenster weichen leicht ab (0,31 statt 0,24 bei g = 0,01), weil die Anfangswerte andere sind; die Aussage „schmales Fenster“ ändert sich nicht.
4. **Exponent der Nullsumme im Quotientenraum:** +9,2·10⁻⁵ / +2,4·10⁻⁴ / +1,9·10⁻⁴ statt +4,9·10⁻⁵ / +1,2·10⁻⁴ / +1,2·10⁻⁴; die Größenordnung (unter 3·10⁻⁴, also positiv, aber weit unter der Chaos-Grenze 0,003, bei 20 000 Schritten) bleibt, die Ziffern hängen am Tangentenstart.
5. **Zählungen der Anfangswerte** (konvergierte Anteile im 3×3-Raster, Randbahnen) unterscheiden sich im Rahmen der Stichprobe von der Vormessung (z. B. Randbahnen 67 / 86 / 170 / 253 statt 62 / 91 / 177 / 271); die Aussagen (kein Chaos im Inneren) sind gleich. Die Prüfung der Randbahnen bei dreifacher Messdauer (Vormessung V9d: nur 63 von 177 bleiben über 0,003) ist in der Messreihe **nicht** wiederholt; eine unabhängige Nachrechnung ließ von 67 / 86 / 170 / 253 Bahnen noch 2 / 0 / 9 / 18 über der Grenze.
6. **Presets und Kennzahlen:** Die Presets laden *alle* Regler (die nicht genannten auf die Voreinstellung), damit das Ergebnis nicht an der Klickreihenfolge hängt.
7. **Periode in Beispiel 1 ab s = 44 (Korrektur nach unabhängiger Prüfung):** Die erste Fassung der Periodenerkennung (20 000 Einschwingschritte, Toleranz 10⁻⁹) meldete bei s = 44,0 bis 59,5 „keine Periode bis 64“ und die App deutete das als „lange Periode oder quasiperiodisch“. Das war ein Messartefakt: Der 2-Zyklus ist dort stabil, sein Multiplikator liegt knapp unter 1 (0,9985 bei s = 44), die Bahn ist nach 20 000 Schritten noch nicht auf 10⁻⁹ an ihm. Die Erkennung prüft jetzt zusätzlich, ob jeder p-te Wert der Bahn streng monoton einem anziehenden Fixpunkt der p-fachen Abbildung zustrebt (`lsc_algorithm.diag_period`), und meldet dann dessen Periode: 2 für alle 104 Werte. Die Reihe `bifurkation` ist mit `tools/precompute_lsc.py bifurkation` neu erzeugt (nur die 32 Perioden bei s = 44,0 bis 59,5 ändern sich von „keine“ zu 2); Beispiel 2 und alle Exponenten bleiben unverändert.

## Ehrliche Grenzen

- **Spielklasse:** 2 Spieler, 2 Kanten, zwei Beispiele (Kosten wie in der Quelle); mehr Spieler oder Kanten sind nicht gemessen. Die Folgerung der Quelle (Chaos für beliebig kleine Schrittweiten bei n Spielern in konstruierten Spielen) ist **nicht** nachgemessen.
- „Chaos nur auf der symmetrischen Diagonale“ heißt hier: In den gemessenen unsymmetrischen Läufen (insgesamt 102 400 Bahnen: 2 400 + 40 000 + 3 × 20 000) fand sich nach Nachprüfung keine chaotische Bahn (im Lauf der zufälligen Spiele waren zwei Bahnen zuerst als chaotisch gezählt, bei 30 000 Einschwingschritten nicht mehr); Chaos wurde nur auf der symmetrischen Diagonale gemessen. Es ist eine Zählung, kein Beweis. Auch Starts nahe der Diagonale (u₂ = u₁ + δ, δ = 10⁻¹² bis 10⁻³, 2 000 Starts, Beispiel 2, s = 26 und 40) enden in einem reinen Gleichgewicht (unabhängige Nachrechnung, nicht in der Messreihe). Die Diagonale ist ein *chaotischer Sattel*: Der Exponent der Querrichtung ist positiv, die Diagonale zieht nicht an.
- Der Befund der exakten Diagonale in Gleitkomma hängt an der **Symmetrie der Rechnung** (beide Spieler führen bitgleich dieselben Operationen aus), nicht an Rundung; mit Rundung hat er nichts zu tun (Verweis auf das Stück zum Gleitkomma-Chaos).
- **Vergessen:** nur die Familie R + e·I, 3×3, ein Parameterfenster (g um 0,02 bis 0,03, s um 0,5), 40 bis 400 Starts je Wert, Fensterkarte gröber als der Hauptbefund; keine Lyapunov-Dimension, kein Beweis, kein Maß des Attraktors. Das Verfahren ist **nicht Hedge**; Vergessen hält die Dualkoordinaten beschränkt, ein innerer Attraktor wird möglich. Hedge mit Vergessen wurde nicht gegen die Literatur geprüft.
- **Nullsumme:** Volumenwachstum im Dualraum ist bei ε ≥ 0,05 nur anfangs vorhanden, bei ε = 0,01 liegt die Bahn am Ende der Messung (10⁵ Schritte) noch im Inneren des Simplex; das stimmt mit dem gelesenen Wortlaut von Cheung/Piliouras überein (Volumenwachstum impliziert keine Instabilität im Primalraum, wenn die Menge nahe am Rand liegt), belegt aber nichts über deren Sätze für sehr kleine Schrittweiten.
- Randbahnen mit λ ≈ 10⁻³ (3×3, reines Hedge) werden nicht gedeutet; die Randgrenze 10⁻⁹ wurde nicht variiert; der Anteil der Spiele oberhalb der Schwelle des gemischten Punkts ist in der Gegenprobe zufälliger Spiele nicht ausgewiesen.
- Zahlen mit Anteilen sind auf 100 bzw. 40 Starts je Wert gezählt (Auflösung 0,01 bzw. 0,025).

## Quellen (Stufen der Lektüre)

- Palaiopanos, Panageas, Piliouras (NeurIPS 2017) **[W: Wortlaut gelesen]**: Variante MWU_ε mit Faktor (1 − ε)^Kosten; bei ε = 1 − e⁻¹⁰ ein Grenzzyklus der Periode 2 (Satz 4.1), Zyklen jeder Länge und Li-Yorke-Chaos in Zwei-Spieler-Lastverteilungsspielen (Satz 4.2), „Chaos for arbitrary small learning rates“ in konstruierten Überlastungsspielen mit n Spielern (Folgerung 4.3 <!-- lint-ok: dezimalpunkt -->). Die Chaos-Aussage betrifft diese **Variante** und diese Spiele, nicht Nullsummenspiele.
- Cheung, Piliouras (arXiv 2005.13996) **[W bis Zeichen 80 000 von 137 000]**: „Lyapunov chaotic“ ist über Volumenwachstum und Lyapunov-Zeit gefasst, nicht als üblicher größter Exponent; die Sätze verlangen sehr kleine Schrittweiten.
- Cheung, Tao (arXiv 2008.00540) **[W bis Zeichen 80 000 von 129 000]**: „fast überall“ bezieht sich auf die Klasse der Spiele (positives Maß im Raum der Bimatrix-Spiele); eine Aussage über fast alle Startpunkte steht im gelesenen Text nicht.
- Sato, Akiyama, Farmer (2002) und Galla, Farmer (2013) zu Chaos beim Lernen mit Gedächtnis: **[G: aus dem Gedächtnis, nicht gelesen]**; nur der Anlass für die Messung mit Vergessen, nirgends als Beleg.

## Offenes

Cheung/Piliouras und Cheung/Tao ab Zeichen 80 000 im Wortlaut lesen; Sato/Akiyama/Farmer und Galla/Farmer lesen oder gestrichen lassen; Spiele mit mehr als zwei Spielern; die Klasse der Spiele von Cheung/Tao messen; Randbahnen mit λ ≈ 10⁻³ deuten; Randgrenze variieren; Maß des Attraktors (Lyapunov-Dimension) bei Vergessen.

## Verifikation

- **Kern:** Handbeispiel (`tests/test_algorithm.py`: Schwellen 8 und 17,78, 2-Zyklus u = 1,776029 bei s = 10, Faktoren, ein Schritt von Hand), Zufallsgenerator gegen Schleifenfassung (`tests/test_rng.py`), Randfälle (`tests/test_edge_cases.py`).
- **Orakel** (`tests/test_oracle_lernen_spiele_chaos.py`): Hedge-Schritt gegen 60-stellige `decimal`-Rechnung; Jacobi-Produkt gegen Complex-Step und finite Differenz; allgemeine m×m-Fassung gegen die Logit-Abbildung; Exponenten gegen Zwei-Bahn-Verfahren und 2D-Tangente; Volumenänderung der Nullsumme gegen die Determinante einer numerischen Jacobi-Matrix.
- **Eigenschaften** (`tests/test_eigenschaften_lernen_spiele_chaos.py`): Kanten und Spieler vertauschen, nur s·a zählt, Konstante auf alle Auszahlungen ändert nichts, Wahrscheinlichkeiten in [0, 1] auch bei s = 80, gleicher Zufalls-Seed gleiche Ausgabe.
- **Claims** (`tests/test_claims.py`) für jede Zahl der Befunde-Tabelle, **Verdikte** je Zweig und je Preset (`tests/test_verdicts.py`), Presets, Layout (Schnellstart oben in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben), AppTest-Smoke (jeder Button, jeder Regler an seinen Grenzen), Footer, Laufzeit-Importe.
- Die Testzahl steht nicht in dieser README; `pytest --collect-only -q tests` zählt sie.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore`, `.streamlit/config.toml` | venv, Caches, Streamlit-Secrets; Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Kernfrage, sechs Experimente mit Verdikt, Grenzen, mathematische Formulierung |
| `lsc_algorithm.py` | Kern 1: Logit-Abbildung des Überlastungsspiels, Diagonale, Exponenten, Zwei-Spieler-Abbildung |
| `lsc_bimatrix.py` | Kern 2: Hedge mit Vergessen in m×m-Spielen, Tangentenverfahren, Zwei-Bahn-Verfahren, Nullsumme |
| `lsc_rng.py` | SplitMix64, vektorisiert |
| `lsc_constants.py`, `lsc_presets.py`, `lsc_layout.py` | Regler-Grenzen und Presets, Permalink, fester Kopf |
| `lsc_evaluation.py`, `lsc_messreihen.py`, `lsc_frozen.py` | Messungen, Verdikte, Messreihen und deren Laden |
| `lsc_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `tools/precompute_lsc.py`, `data/lsc_vorgerechnet.json` | Skript, das die Messreihen reproduziert, und deren Ergebnis |
| `requirements-dev.txt`, `requirements.txt` | Laufzeit plus pytest; Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Handbeispiel, Orakel, Eigenschaften, Claims, Verdikte, Randfälle, Presets, Layout, AppTest-Smoke, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **PDF-Export:** Die Demo hat kein eigenes Ergebnis, das ein Bericht festhalten müsste; die Zahlen stehen in README und Messdatei.
- **Mehr als zwei Spieler, mehr als zwei Kanten** und die Klasse der Spiele von Cheung/Tao: nicht gemessen (siehe Offenes).
- **Eigene Kosten je Kante im Regler:** Die zwei Beispiele aus der Quelle sind fest; die Kostenabweichung von Spieler 2 steht in Experiment 3.
- **Mutationstests:** gibt es im Portfolio nicht mehr; Orakel-, Claims- und Eigenschaftstests bleiben.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Die Messreihen neu rechnen (einige Minuten): `python tools/precompute_lsc.py`.

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
