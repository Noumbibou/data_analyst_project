# Etape pour creer un fichier exécutable de l'Application
## Etape 1 : Installation
pip install pyinstaller

## Etape 2 : Pour créer un seul fichier exécutable
pyinstaller --onefile traduction_anglais_project_fin.py

## Etape 3 : Pour ne pas afficher la console (applications GUI)
pyinstaller --onefile --noconsole traduction_anglais_project_fin.py

## Etape 4 : Pour ajouter une icône
pyinstaller --onefile --icon=mon_icone.ico traduction_anglais_project_fin.py