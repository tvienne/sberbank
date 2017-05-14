Kaggle Sberbank :

==============
Comment lancer le projet :

1 - Téléchargez depuis Kaggle les fichiers suivants et placez-les dans le dossier data :
- data-dictionnary.txt
- macro.csv
- sample_submission.txt
- test.csv
- train.csv

2 - Dans votre fichier main, ne pas oublier d'inclure les lignes suivantes afin que python arrive à trouver les
répertoires :

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)



