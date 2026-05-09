# Workflow gitowy dla notebookow Jupyter

Notebook `.ipynb` jest plikiem JSON. Po zwyklym uruchomieniu Jupyter zapisuje
w nim nie tylko kod i markdown, ale tez:

- outputy komorek, w tym mapy jako ogromne obrazki base64,
- `execution_count`,
- metadane kernela i lokalnego srodowiska,
- stan widgetow,
- czasem sciezki lokalne z komputera osoby, ktora uruchamiala notebook.

To dlatego konflikty merge wygladaja dramatycznie. Dwie osoby moga nie zmienic
zadnej linijki kodu, a mimo tego Git widzi tysiace zmienionych linii.

## Zasada projektu

Do repo commitujemy czysty notebook: kod, markdown i parametry. Nie commitujemy
outputow komorek.

Wygenerowane wersje do pokazania trzymaj w `outputs/`, ktory jest ignorowany
przez Git.

## Przed commitem

Uruchom:

```powershell
py -3.11 scripts\clean_notebooks.py
```

Potem sprawdz diff:

```powershell
git diff -- notebooks
```

W diffie powinny zostac tylko realne zmiany w kodzie albo opisie, a nie
`execution_count`, lokalne sciezki, PNG/base64 i outputy.

## Wygenerowanie wersji do prezentacji

Czysty notebook mozna wykonac i zapisac jako artefakt:

```powershell
py -3.11 -m nbconvert --to notebook --execute notebooks\01_wroclaw_metro_planner.ipynb --output ..\outputs\01_wroclaw_metro_planner_executed.ipynb --ExecutePreprocessor.timeout=900
```

Ten plik jest do ogladania i prezentacji, nie do commitowania.

## Opcjonalnie: automatyczny pre-commit

Mozna wlaczyc hook, ktory czysci notebooki automatycznie przed commitem:

```powershell
py -3.11 -m pip install -r requirements-dev.txt
py -3.11 -m pre_commit install
```

Od tej chwili `git commit` sam uruchomi `scripts\clean_notebooks.py`.

## Gdy dwie osoby pracuja naraz

Najbezpieczniejszy schemat:

1. Przed praca: `git pull`.
2. Zmieniaj kod albo opis w notebooku.
3. Przed commitem: `py -3.11 scripts\clean_notebooks.py`.
4. Sprawdz: `git diff -- notebooks`.
5. Commituj tylko czysty notebook.
6. Outputy i wygenerowane mapy trzymaj w `outputs/`.

## Dodatkowe narzedzie do konfliktow

Do trudniejszych merge'y notebookow warto uzyc `nbdime`:

```powershell
py -3.11 -m pip install -r requirements-dev.txt
py -3.11 -m nbdime config-git --enable --global
```

To nie zastapi czyszczenia outputow, ale pokazuje konflikty notebookow w bardziej
czytelny sposob niz zwykly Git.
