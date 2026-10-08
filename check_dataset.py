
import os

for dossier_principal in ['asl_alphabet_train/asl_alphabet_train', 'asl_alphabet_test/asl_alphabet_test']:
    print(f'\n📁 {dossier_principal}')
    print('='*40)
    chemin = f'./{dossier_principal}'
    if not os.path.exists(chemin):
        print('❌ Dossier introuvable !')
        continue
    sous_dossiers = sorted(os.listdir(chemin))
    print(f'Nombre de classes : {len(sous_dossiers)}')
    print(f'Classes : {sous_dossiers}')
    print()
    for s in sous_dossiers:
        sp = os.path.join(chemin, s)
        if os.path.isdir(sp):
            nb = len(os.listdir(sp))
            print(f'  {s} → {nb} images')
