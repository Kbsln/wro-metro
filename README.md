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
- warstwa geologiczna/kosztowa moze byc dodana jako `data/raw/geology.geojson` lub `data/raw/cost_zones.geojson`; kolumna `cost_factor` powyzej `1.0` naklada kare za trudny teren,
- rzeka i wody powierzchniowe nie sa zakazem dla metra; sa bariera komunikacyjna, ktorej przecięcie moze poprawic siec,
- regionalne centra popytu sa klastrami obszarow demograficznych, a nie recznie wpisanymi punktami,
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

Notebook automatycznie uzyje `data/raw/dem-rejurb-rejstat-shp.zip`, jesli plik istnieje. Jezeli dodasz `data/raw/flood_zones.geojson`, zostanie uzyty jako wlasciwa warstwa zalewowa. `data/raw/geology.geojson` mozna wygenerowac z publicznej uslugi PIG-PIB/CBDG Mapa Litogenetyczna Polski 1:50 000:

```powershell
py -3.11 scripts\download_geology_data.py
```

Warstwa geologiczna zawiera `cost_factor` jako heurystyczny mnoznik trudnosci budowy wyprowadzony z litologii MLP50k. Model bedzie karal trasy przez drozszy teren. `wody-powierzchniowe.zip` jest uzywane jako warstwa rzek/wody do oceny przeciec komunikacyjnych, nie jako zakaz budowy metra.

## Uruchomienie

```powershell
py -3.11 -m pip install -r requirements.txt
py -3.11 -m jupyter lab
```

Potem otworz `notebooks/01_wroclaw_metro_planner.ipynb`.
