import os
import urllib.request
import ssl

def download_identity_file():
    # A known GitHub mirror for identity_CelebA.txt (often used in tutorials when GDrive fails)
    # Trying a few common raw URLs for it
    urls = [
        "https://raw.githubusercontent.com/BUPT-Tencent/CelebA-Spoof/master/data/identity_CelebA.txt",
        "https://raw.githubusercontent.com/yashsmehta/celeba-dataset/master/identity_CelebA.txt",
        "https://raw.githubusercontent.com/natanielruiz/deep-head-pose/master/data/identity_CelebA.txt"
    ]
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    output_path = "c:/Users/root/project/federetad_learning/identity_CelebA.txt"
    if os.path.exists(output_path):
        print(f"File already exists at {output_path}")
        return True

    for url in urls:
        try:
            print(f"Trying to download from {url}...")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, context=ctx) as response, open(output_path, 'wb') as out_file:
                data = response.read()
                out_file.write(data)
            print(f"Successfully downloaded to {output_path}")
            return True
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            
    print("Could not download identity_CelebA.txt from mirror URLs. Trying gdown...")
    try:
        import gdown
        file_id = '1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output_path, quiet=False)
        print("Downloaded via gdown.")
        return True
    except Exception as e:
        print(f"Failed with gdown: {e}")
        
    return False

if __name__ == "__main__":
    download_identity_file()
