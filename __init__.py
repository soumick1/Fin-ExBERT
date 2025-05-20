import subprocess
import sys

def install_without_version(package_name):
    package_name = str(package_name)
    try:
        __import__(package_name)
        print(f"{package_name} is already installed.")
    except ImportError:
        print(f"{package_name} is not installed. Installing now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", '-U', package_name])


def install_with_version(package_name, package_version):
    package_name = str(package_name)
    package_version = str(package_version)
    try:
        pkg = __import__(package_name)
        installed_version = pkg.__version__
        if installed_version == package_version:
            print(f"{package_name} {package_version} is already installed.")
        else:
            print(f"{package_name} {installed_version} is installed, but {package_version} is required. Updating now...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package_name}=={package_version}"])
    except ImportError:
        print(f"Installing version {package_version} now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package_name}=={package_version}"])

if __name__=='__main__':
    packages = [['torch', ''], ['datasets', ''], ['spacy', ''], ['networkx', ''], ['numpy', '1.26.4']]
    for package in packages:
        if package[1] == '':
            install_without_version(package[0])
        else:
            install_with_version(package[0], package[1])

    subprocess.check_call([sys.executable, "python", "-m", "spacy", "download", "en_core_web_sm"])
    
