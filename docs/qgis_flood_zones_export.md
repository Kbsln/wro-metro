# Eksport obszarow zalewowych z QGIS do projektu

Ta instrukcja opisuje, jak przygotowac warstwe `data/raw/flood_zones.geojson`,
ktora notebook `01_wroclaw_metro_planner.ipynb` automatycznie wczyta jako
obszary zagrozenia powodziowego.

Jesli QGIS sprawia problem, najprostsza sciezka w tym projekcie to pobranie
danych z API/WFS skryptem:

```powershell
py -3.11 scripts\download_flood_zones_isok.py
```

Skrypt korzysta z oficjalnej uslugi WFS ISOK/Wod Polskich, pobiera
`nz-core:HazardArea`, przycina wynik do granicy Wroclawia i zapisuje
`data/raw/flood_zones.geojson`.

Najwazniejsze rozroznienie: warstwa `wody-powierzchniowe.zip` z SIP Wroclaw
pokazuje rzeki, kanaly i zbiorniki. To nie jest mapa zalewowa. Do obszarow
zalewowych uzywamy MZP, czyli Map Zagrozenia Powodziowego z Wod Polskich /
ISOK / Hydroportalu.

## Co pobrac

Do modelu metra potrzebujemy poligonow, nie obrazka z WMS.

Minimum:

- `MZP - obszary zagrozenia powodziowego, prawdopodobienstwo 1% (Q1%)`

Lepszy wariant do opisu projektu:

- `Q10%` - wysokie prawdopodobienstwo, powodz raz na ok. 10 lat,
- `Q1%` - srednie prawdopodobienstwo, powodz raz na ok. 100 lat,
- `Q0.2%` - niskie prawdopodobienstwo, scenariusz ekstremalny, raz na ok. 500 lat.

W modelu mozna potem przyjac np. Q10 i Q1 jako mocna kare lokalizacji, a Q0.2
jako slabsza kare lub wariant wrazliwosci.

## 1. Instalacja QGIS

1. Zainstaluj QGIS LTR ze strony QGIS.
2. Uruchom QGIS.
3. W prawym dolnym rogu ustaw CRS projektu na `EPSG:2180` albo `EPSG:2177`.
   Nie jest to krytyczne, bo notebook i tak przelicza uklady wspolrzednych,
   ale w Polsce te uklady sa wygodniejsze do analiz metrowych niz WGS84.

## 2. Instalacja wtyczki Wod Polskich

1. W QGIS wejdz w `Wtyczki` -> `Zarzadzaj i instaluj wtyczki`.
2. Wyszukaj `Wody Polskie - Baza WMS`.
3. Kliknij `Instaluj wtyczke`.
4. Po instalacji powinna pojawic sie ikona lub pozycja menu wtyczki Wod Polskich.

Wtyczka ulatwia dodanie uslug Wod Polskich, w tym MZP i MRP. Jezeli widzisz
tylko warstwe WMS, traktuj ja jako podglad. Do naszego modelu potrzebny jest
eksport/pobranie danych wektorowych, czyli shapefile, GeoPackage albo GeoJSON.

## 3. Wczytanie granicy Wroclawia

Najprosciej wykorzystac granice osiedli, ktore juz pobiera skrypt projektu.

1. Upewnij sie, ze masz dane:

   ```powershell
   py -3.11 scripts\download_wroclaw_data.py
   ```

2. W QGIS wybierz `Warstwa` -> `Dodaj warstwe` -> `Dodaj warstwe wektorowa`.
3. Jako zrodlo wskaz `data/raw/granice-osiedli.zip`.
4. Dodaj warstwe osiedli do projektu.
5. Utworz jedna granice miasta:
   - wejdz w `Wektor` -> `Narzędzia geoprzetwarzania` -> `Rozpusc` (`Dissolve`),
   - `Warstwa wejsciowa`: granice osiedli,
   - nie wybieraj pola rozpuszczania, czyli rozpusc wszystkie obiekty,
   - zapisz wynik jako `data/processed/wroclaw_boundary.gpkg`.

Ta warstwa bedzie maska do przyciecia MZP tylko do Wroclawia.

## 4. Dodanie lub pobranie MZP

1. Otworz wtyczke `Wody Polskie - Baza WMS`.
2. Wybierz grupe `Mapy zagrozenia powodziowego (MZP)`.
3. Dodaj lub pobierz warstwy z obszarami zagrozenia powodziowego:
   - Q1% jako minimum,
   - opcjonalnie Q10% i Q0.2%.
4. Sprawdz, czy warstwa jest poligonowa:
   - jezeli w panelu warstw ma typ wektorowy/poligonowy, mozna ja ciac i eksportowac,
   - jezeli jest WMS/raster, to jest tylko obraz do podgladu. Wtedy uzyj opcji
     pobierania danych przestrzennych we wtyczce albo pobierz paczke wektorowa
     z Hydroportalu/SIGW.

### Jesli MZP widac na mapie, ale nie da sie jej wybrac w `Clip`

To prawie zawsze oznacza, ze dodana warstwa jest WMS/WMTS, czyli obraz mapy.
Jest widoczna w projekcie, ale QGIS nie pokazuje jej w narzedziach wektorowych,
bo nie ma tam obiektow typu `Polygon`.

Wtedy dodaj MZP jako WFS:

1. Wejdz w `Warstwa` -> `Menedzer zrodel danych`.
2. Wybierz `WFS / OGC API - Features`.
3. Kliknij `Nowy`.
4. Nazwa: `ISOK MZP MRP WFS`.
5. URL:

   ```text
   https://wody.isok.gov.pl/wss/INSPIRE/INSPIRE_NZ_HY_MZPMRP_WFS?REQUEST=GetCapabilities&SERVICE=WFS&VERSION=2.0.0
   ```

6. Kliknij `OK`, potem `Polacz`.
7. Z listy warstw wybierz najpierw `nz-core:HazardArea`. To jest najlepszy
   kandydat dla Map Zagrozenia Powodziowego.
8. Jezeli QGIS daje taka opcje, zaznacz pobieranie tylko obiektow z biezacego
   zakresu mapy. Najpierw przybliz widok na Wroclaw, bo krajowa usluga WFS moze
   byc wolna.
9. Dodaj warstwe do projektu.

Po dodaniu WFS sprawdz:

- prawy klik na warstwe -> `Otworz tabele atrybutow`,
- prawy klik -> `Wlasciwosci` -> `Informacje`,
- typ geometrii powinien byc poligonowy.

Dopiero taka warstwa pojawi sie w narzedziu `Przytnij` jako `Warstwa
wejsciowa`.

## 5. Przyciecie MZP do Wroclawia

Dla kazdej warstwy MZP, ktora chcesz wykorzystac:

1. Wejdz w `Wektor` -> `Narzędzia geoprzetwarzania` -> `Przytnij` (`Clip`).
2. `Warstwa wejsciowa`: warstwa MZP, np. Q1%.
3. `Warstwa nakladki`: `wroclaw_boundary`.
4. `Przyciete`: zapisz do pliku tymczasowego albo do GeoPackage.
5. Uruchom.

Jezeli QGIS ma problem z geometriami:

1. Wejdz w `Wektor` -> `Narzędzia geometrii` -> `Napraw geometrie`
   (`Fix geometries`).
2. Jako wejscie wybierz przycieta warstwe MZP.
3. Wynik wykorzystaj do eksportu.

## 6. Opcjonalne polaczenie kilku scenariuszy Q

Jezeli pobrales Q10, Q1 i Q0.2, warto je polaczyc w jedna warstwe.

1. Dla kazdej przycietej warstwy dodaj pole:
   - otworz tabele atrybutow,
   - kliknij `Kalkulator pol`,
   - zaznacz `Utworz nowe pole`,
   - nazwa pola: `scenario`,
   - typ: tekst,
   - wartosc: odpowiednio `'Q10'`, `'Q1'` albo `'Q02'`.
2. Dodaj drugie pole `probability` typu liczbowego:
   - Q10: `10`,
   - Q1: `1`,
   - Q02: `0.2`.
3. Wejdz w `Wektor` -> `Zarzadzanie danymi` -> `Polacz warstwy wektorowe`
   (`Merge Vector Layers`).
4. Jako wejscie wybierz przyciete warstwy Q10/Q1/Q0.2.

Jesli chcesz zrobic szybko pierwsza wersje projektu, wystarczy samo Q1 bez
laczenia.

## 7. Eksport do pliku, ktory czyta notebook

1. Kliknij prawym przyciskiem gotowa warstwe MZP.
2. Wybierz `Eksportuj` -> `Zapisz obiekty jako...`.
3. Ustaw:
   - `Format`: `GeoJSON`,
   - `Nazwa pliku`: `C:\Users\sila6\Code\wro-metro\data\raw\flood_zones.geojson`,
   - `CRS`: `EPSG:4326 - WGS 84` albo zostaw CRS warstwy, jesli QGIS poprawnie go zapisuje,
   - `Kodowanie`: `UTF-8`,
   - geometria: poligon/multipoligon; jezeli jest opcja `Force multi-type`, zaznacz ja.
4. Kliknij `OK`.

Notebook szuka tego pliku automatycznie. Alternatywnie mozesz zapisac:

- `data/raw/flood_zones.gpkg`,
- `data/raw/flood_zones.zip`,
- albo rozpakowany shapefile w `data/raw/flood_zones/`.

Najmniej problemow sprawia jednak pojedynczy `flood_zones.geojson`.

## 8. Szybka kontrola po eksporcie

W terminalu projektu uruchom:

```powershell
py -3.11 -c "import geopandas as gpd; gdf=gpd.read_file('data/raw/flood_zones.geojson'); print(len(gdf), gdf.crs, gdf.geom_type.value_counts().to_dict())"
```

Oczekujesz:

- liczba obiektow wieksza niz `0`,
- CRS nie jest `None`,
- typ geometrii to `Polygon` albo `MultiPolygon`.

Potem uruchom notebook ponownie. W sekcji ladowania danych komunikat o
obszarach zalewowych powinien przestac mowic, ze brakuje lokalnej warstwy MZP.

## Typowe problemy

- Wynikiem przyciecia jest ksztalt miasta zamiast terenow zalewowych: zwykle
  jako `Warstwe wejsciowa` wybrano granice Wroclawia. W narzedziu `Clip`
  ustaw odwrotnie: `Warstwa wejsciowa` = MZP/tereny zalewowe, `Warstwa
  nakladki` = granica Wroclawia. QGIS przycina zawsze warstwe wejsciowa
  ksztaltem warstwy nakladki.
- Plik po eksporcie jest pusty: najczesciej przycieto zla warstwe, warstwa MZP
  nie obejmowala Wroclawia albo uzyto samego WMS zamiast danych wektorowych.
- QGIS pozwala zapisac tylko obraz: to znaczy, ze masz warstwe WMS/raster.
  Potrzebujesz pobrania danych wektorowych.
- Warstwa z wtyczki Wod Polskich nie ma tabeli atrybutow: to najpewniej WMS,
  czyli obraz mapy. Dla modelu potrzebujesz warstwy wektorowej z poligonami
  MZP. Warstwa wektorowa powinna pozwalac otworzyc `Tabele atrybutow`,
  miec liczbe obiektow wieksza niz 0 i typ geometrii `Polygon` lub
  `MultiPolygon`.
- MZP widac w panelu warstw, ale nie ma jej na liscie w `Clip`: to tez oznacza,
  ze QGIS widzi ja jako raster/WMS, a nie jako wektor. Dodaj ja przez WFS albo
  pobierz dane przestrzenne i zapisz lokalnie jako GPKG/GeoJSON.
- Notebook dalej nie widzi danych: sprawdz dokladna nazwe i lokalizacje pliku:
  `data/raw/flood_zones.geojson`.
- Mapa w notebooku wyglada dziwnie przesunieta: sprawdz, czy eksport mial
  poprawny CRS. Najbezpieczniej zapisac GeoJSON jako `EPSG:4326`.
- QGIS jest wolny: zapisuj roboczo do GeoPackage (`.gpkg`), a dopiero finalny
  wynik eksportuj do GeoJSON.

## Najprostsza sciezka awaryjna

Jezeli `Clip` dalej robi zly wynik, uzyj selekcji przestrzennej zamiast
przycinania. Do modelu lepiej miec lekko nadmiarowe poligony MZP niz przypadkowo
wyeksportowac sama granice miasta.

1. Wejdz w `Wektor` -> `Narzędzia badawcze` -> `Wybierz wedlug lokalizacji`
   (`Select by Location`).
2. `Wybierz obiekty z`: warstwa MZP/tereny zalewowe.
3. Predykat: `przecina` (`intersects`).
4. `Porownaj z obiektami z`: `wroclaw_boundary`.
5. Uruchom.
6. Kliknij prawym na warstwe MZP -> `Eksportuj` -> `Zapisz zaznaczone obiekty
   jako...`.
7. Zapisz jako `data/raw/flood_zones.geojson`.

Po takim eksporcie mozesz jeszcze raz odpalic `Clip`, ale juz na zapisanej
lokalnej warstwie GeoJSON. Zwykle lokalny GeoJSON/GPKG sprawia mniej problemow
niz bezposrednia praca na warstwie z uslugi sieciowej.

## Zrodla

- MZP/MRP, definicje i zakres: https://powodz.gov.pl/pl/o_mapach
- Hydroportal Wod Polskich: https://wody.isok.gov.pl/hydroportal.html
- Wtyczka QGIS `Wody Polskie - Baza WMS`: https://plugins.qgis.org/plugins/wody_polskie_wms/
- Dokumentacja QGIS, wtyczki: https://docs.qgis.org/latest/en/docs/user_manual/plugins/plugins.html
- Dokumentacja QGIS, Clip i Extract/clip by extent: https://docs.qgis.org/latest/en/docs/user_manual/processing_algs/qgis/vectoroverlay.html
