import os
import nbformat
import re
import pkg_resources
import sys
import stdlib_list

IMPORT_TO_PYPI = {
    'bs4': 'beautifulsoup4',
    'skopt': 'scikit-optimize',
    'cv2': 'opencv-python',
    'PIL': 'Pillow',
    'yaml': 'PyYAML',
    'sklearn': 'scikit-learn',
    'bayes_opt': 'bayesian-optimization',
}

def find_notebooks(root_dir):
    notebooks = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".ipynb"):
                notebooks.append(os.path.join(dirpath, file))
    return notebooks

def extract_imports_from_notebook(notebook_path):
    imports = set()
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        for cell in nb.cells:
            if cell.cell_type == 'code':
                lines = cell.source.split('\n')
                for line in lines:
                    if re.match(r'^\s*(import|from)\s+[a-zA-Z0-9_\.]+', line):
                        imports.add(line.strip())
    return imports

def collect_all_imports(root_dir):
    all_imports = set()
    notebooks = find_notebooks(root_dir)
    for nb_path in notebooks:
        imports = extract_imports_from_notebook(nb_path)
        all_imports.update(imports)
    return all_imports

def save_imports_to_file(imports, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Toplanan tüm importlar\n")
        for imp in sorted(imports):
            f.write(imp + '\n')

py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
std_libs = set(stdlib_list.stdlib_list(py_version))

root_folder = "."
output_file = os.path.join(root_folder, "all_imports.py")

imports = collect_all_imports(root_folder)
save_imports_to_file(imports, output_file)

print(f"{len(imports)} import satırı {output_file} dosyasına yazıldı.")

imports = set()
with open("all_imports.py", "r", encoding="utf-8") as f:
    for line in f:
        match = re.match(r"^\s*(?:import|from)\s+([a-zA-Z0-9_\.]+)", line)
        if match:
            pkg = match.group(1).split('.')[0]
            imports.add(pkg)

with open("requirements.txt", "w", encoding="utf-8") as f:
    for pkg in sorted(imports):
        pypi_name = IMPORT_TO_PYPI.get(pkg, pkg)
        if pypi_name not in std_libs:
            try:
                version = pkg_resources.get_distribution(pypi_name).version
                f.write(f"{pypi_name}=={version}\n")
            except pkg_resources.DistributionNotFound:
                f.write(f"{pypi_name}\n")