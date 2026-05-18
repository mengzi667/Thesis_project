from pathlib import Path
p=Path(r'D:\TUD\Thesis_project\results\pdf_annotations_SM-Yumeng_Thesis_midterm.txt')
text=p.read_text(encoding='utf-8',errors='ignore').splitlines()
page=None
items=[]
for ln in text:
    if ln.startswith('=== Page '):
        try:
            page=int(ln.split()[2])
        except:
            page=None
    elif ln.startswith('    '):
        s=ln.strip()
        if s and s!='(no text)':
            items.append((page,s))
print('total_text_comments',len(items))
for pg,s in items[:180]:
    print(f'{pg}\t{s}')
