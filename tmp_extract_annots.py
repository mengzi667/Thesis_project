from pypdf import PdfReader
from pathlib import Path

pdf_path = Path(r"C:\Users\wxdy\Downloads\SM-Yumeng_Thesis_midterm.pdf")
out_path = Path(r"D:\TUD\Thesis_project\results\pdf_annotations_SM-Yumeng_Thesis_midterm.txt")
reader = PdfReader(str(pdf_path))
lines=[]
count=0
for pi,page in enumerate(reader.pages, start=1):
    annots = page.get('/Annots')
    if not annots:
        continue
    lines.append(f"=== Page {pi} ===")
    for ai,a in enumerate(annots, start=1):
        try:
            obj = a.get_object()
        except Exception:
            continue
        subtype = str(obj.get('/Subtype',''))
        author = obj.get('/T','')
        contents = obj.get('/Contents','')
        subject = obj.get('/Subj','')
        if hasattr(author,'get_object'):
            author = author.get_object()
        if hasattr(contents,'get_object'):
            contents = contents.get_object()
        if hasattr(subject,'get_object'):
            subject = subject.get_object()
        text = str(contents).strip()
        subj = str(subject).strip()
        auth = str(author).strip()
        if text or subj:
            count += 1
            lines.append(f"[{ai}] subtype={subtype} author={auth} subject={subj}")
            lines.append(f"    {text}")
        else:
            # keep non-text markups noted
            count += 1
            lines.append(f"[{ai}] subtype={subtype} author={auth} subject={subj} (no text)")
    lines.append("")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\n".join(lines), encoding='utf-8')
print(f"pages={len(reader.pages)} annotations={count}")
print(str(out_path))
