import subprocess
import os

os.chdir('report')
print('Running pdflatex pass 1...')
res1 = subprocess.run(['pdflatex', '-interaction=nonstopmode', 'main.tex'], capture_output=True, text=True)
print('pdflatex 1 code:', res1.returncode)

print('Running bibtex...')
res_bib = subprocess.run(['bibtex', 'main'], capture_output=True, text=True)
print('bibtex code:', res_bib.returncode)

print('Running pdflatex pass 2...')
res2 = subprocess.run(['pdflatex', '-interaction=nonstopmode', 'main.tex'], capture_output=True, text=True)
print('pdflatex 2 code:', res2.returncode)

print('Running pdflatex pass 3...')
res3 = subprocess.run(['pdflatex', '-interaction=nonstopmode', 'main.tex'], capture_output=True, text=True)
print('pdflatex 3 code:', res3.returncode)

lines = res3.stdout.splitlines()
errors = [line for line in lines if line.startswith('!')]
warnings = [line for line in lines if 'Warning' in line]
overfull = [line for line in lines if 'Overfull \\hbox' in line]

print('\n=== SUMMARY ===')
print(f'Errors: {len(errors)}')
for e in errors[:5]:
    print('  ', e)
print(f'Warnings: {len(warnings)}')
for w in warnings[:5]:
    print('  ', w)
print(f'Overfull boxes: {len(overfull)}')
for o in overfull:
    print('  ', o)

import pymupdf
doc = pymupdf.open('main.pdf')
print(f'\nTOTAL PAGES: {len(doc)}')
