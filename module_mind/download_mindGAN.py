import dlutils
import zipfile
# https://drive.google.com/file/d/17ZUtqR6nYw1pUxzKKyb5EJUTZF7s1SYW
dlutils.download.from_google_drive('17ZUtqR6nYw1pUxzKKyb5EJUTZF7s1SYW', directory='module_mind/')

with zipfile.ZipFile('module_mind/output.zip', 'r') as zip_ref:
    zip_ref.extractall('module_mind/')
