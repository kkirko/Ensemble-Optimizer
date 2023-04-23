import sys
import pkg_resources

# Libraries list
lib_list = [
    'numpy',
    'scikit-learn',
    'lightgbm',
    'catboost',
    'torch',
    'matplotlib'
]

# Extract libraries versions
installed_packages = {package.key: package.version for package in pkg_resources.working_set}
used_packages = {library: installed_packages[library] for library in lib_list}

# Save python and libraries versions to file
with open("requirements.txt", "w") as f:
    f.write(f"Python=={sys.version.split()[0]}\n")
    for library, version in used_packages.items():
        f.write(f"{library}=={version}\n")