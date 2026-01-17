# Etape pour creer un fichier exécutable de l'Application
# Etape 1 : Installation
pip install pyinstaller

# Etape 2 : Commande de base (crée un dossier avec tous les fichiers)
pyinstaller votre_script.py

# Etape 3 : Pour créer un seul fichier exécutable
pyinstaller --onefile votre_script.py

# Etape 4 : Pour ne pas afficher la console (applications GUI)
pyinstaller --onefile --noconsole votre_script.py

# Etape 5 : Pour ajouter une icône
pyinstaller --onefile --icon=mon_icone.ico votre_script.py