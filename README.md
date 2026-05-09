# Wroclaw Metro Planner

Eksploracyjny projekt do planowania wariantow metra we Wroclawiu. Punkt startowy jest w notebooku:

`notebooks/01_wroclaw_metro_planner.ipynb`

Model bazowy:

- dlugosc jednej linii: 23,1 km, czyli benchmark I linii metra w Warszawie,
- liczba stacji: 21,
- trasa musi przechodzic przez centrum,
- centrum jest modelowane jako Stare Miasto, bo laczy funkcje centralne i turystyczne,
- popyt jest rysowany jako obszary demograficzne, a wazone punkty sa tylko techniczna reprezentacja obszarow w algorytmie,
- strefy zalewowe sa traktowane jako obszary ryzyka, a popyt lezacy w nich jest przesuwany do najblizszego wolnego punktu,
- warstwa zalewowa nie jest generowana z rzek; trzeba dodac realne MZP/ISOK jako `data/raw/flood_zones.geojson`,
- warstwa geologiczna/kosztowa jest wczytywana z `data/raw/geology.geojson` albo `data/raw/cost_zones.*`; kolumna `cost_factor` powyzej `1.0` podnosi koszt i kare za trudniejszy teren, a `cost_factor >= 1.5` jest traktowany jako wysokie ryzyko geologiczne,
- rzeka i wody powierzchniowe nie sa zakazem dla metra; sa bariera komunikacyjna, ktorej przecięcie moze poprawic siec,
- stacje i kotwice unikaja wod powierzchniowych oraz buforow MZP, nawet gdy sam tunel moze przeciac rzeke,
- regionalne centra popytu sa klastrami obszarow demograficznych, a nie recznie wpisanymi punktami,
- kandydaci na kotwice maja minimalna reprezentacje sektorow miasta, zeby poludnie/wschod/zachod/polnoc nie znikaly przy dominacji centrum,
- kandydaci na kotwice stacji maja wage z lokalnego zasiegu dojscia, dzieki czemu okolice centrum nie sa redukowane do jednej wymuszonej kropki,
- kotwice stacji lezace w MZP albo w buforze mozliwych podtopien sa przesuwane do najblizszego miejsca poza strefa ryzyka,
- glowny algorytm to orienteering / prize-collecting TSP: komiwojazer z nagrodami i limitem dlugosci, czyli problem NP-trudny laczacy idee TSP i plecaka,
- heurystyka wybiera kandydatow na stacje metoda greedy insertion, poprawia kolejnosc przez 2-opt i ocenia warianty funkcja celu,
- dla malej probki kandydatow jest tez solver dokladny brute force, zeby pokazac eksplozje kombinatoryczna i porownac heurystyke z optimum,
- kolejne linie uwzgledniaja popyt resztkowy, premie za przeciecia/przesiadki oraz kare za nakladanie sie na juz zaplanowany korytarz,
- scenariusze 1, 2 i 3 oznaczaja odpowiednio jedna, dwie i trzy linie.

## Dane

Najlepszy start to dane SIP Wroclawia: demografia 1998-2025, granice osiedli, adresy i warstwy bazowe. Do ryzyka powodziowego uzyj Hydroportalu ISOK/Wod Polskich. Frekwencje i lokale wyborcze mozna potraktowac jako alternatywny proxy popytu, ale dla pracy inzynierskiej/magisterskiej lepiej oprzec finalny popyt na demografii SIP albo GUS, a glosy potraktowac jako walidacje aktywnosci dziennej.

Oficjalne paczki SIP mozna pobrac skryptem:

```powershell
py -3.11 scripts\download_wroclaw_data.py
```

Obszary zalewowe MZP mozna pobrac bez QGIS bezposrednio z oficjalnej uslugi
WFS ISOK/Wod Polskich:

```powershell
py -3.11 scripts\download_flood_zones_isok.py
```

Skrypt zapisuje wynik do `data/raw/flood_zones.geojson`, czyli do pliku, ktory
notebook automatycznie wczytuje jako wlasciwa warstwe zalewowa. `wody-powierzchniowe.zip`
jest uzywane jako warstwa rzek/wody do oceny przeciec komunikacyjnych, nie jako
zakaz budowy metra.

Warstwe geologiczna/kosztowa dla Wroclawia mozna odtworzyc z publicznej uslugi
PIG-PIB/CBDG Mapa Litogenetyczna Polski 1:50 000:

```powershell
py -3.11 scripts\download_geology_data.py
```

Skrypt zapisuje `data/raw/geology.geojson`. Pole `cost_factor` jest heurystyka
trudnosci budowy wyprowadzona z litologii. Solver przelicza je na
`geology_excess_km`: np. 1 km przez obszar `cost_factor = 1.35` daje 0,35 km
nadwyzki geologicznej, a ta nadwyzka jest karana przez `geology_penalty_per_km`.
Dodatkowo trudna geologia obniza atrakcyjnosc kandydatow na kotwice i stacje,
a odcinki `cost_factor >= 1.5` dostaja osobna kare `high_geology_penalty_per_km`.
Algorytm karze tez zbyt objazdowe i cofajace sie korytarze, zeby ograniczyc
efekt "TSP po punktach" i promowac naturalniejsze ramiona linii.
Kolejne linie mocniej wygaszaja popyt w obszarach juz obsluzonych i dostaja
wyzsza kare za przebieg w szerszym buforze poprzednich linii, co zmniejsza
ryzyko planowania dwoch nitek w ten sam rejon miasta.

## Uruchomienie

```powershell
py -3.11 -m pip install -r requirements.txt
py -3.11 -m jupyter lab
```

Potem otworz `notebooks/01_wroclaw_metro_planner.ipynb`.

## Praca zespolowa z notebookami

Przed commitem czysc outputy i lokalne metadane notebooka:

```powershell
py -3.11 scripts\clean_notebooks.py
```

Szczegoly sa w `docs/git_notebook_workflow.md`. To ogranicza konflikty merge,
bo do Git trafia kod i opis, a nie lokalne sciezki, execution county i obrazki
base64 z map.
