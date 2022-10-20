from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

def authenticate_Gdrive(): 
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)
    return drive

def download_Gdrive_folder(drive, Gid):
    folder_id = "'" + Gid +"' in parents"
    file_list = drive.ListFile({'q': folder_id}).GetList()

    for f in file_list:
        # 3. Create & download by id.
        if not f['labels']['trashed']:
            print('title: %s, id: %s' % (f['title'], f['id']))
            f_ = drive.CreateFile({'id': f['id']})
            f_.GetContentFile(f['title'])
            
def download_file_from_Gdrive(drive, name, Gid):
    downloaded = drive.CreateFile({'id': Gid})   # replace the id with id of file you want to access
    downloaded.GetContentFile(name)        # replace the file name with your file

# To work properly this is sometimes needed: !pip install -q httplib2==0.15.0 # only relevant if files has to be pushed to google drive
def upload_file_to_Gdrive(drive, name, parent_folder_Gid): 
    zip_file = drive.CreateFile({'title':name,'parents': [{'id':parent_folder_Gid}]})
    zip_file.SetContentFile(name)
    zip_file.Upload()
