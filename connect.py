from pywinauto import application
import os
import time


def auto_connect(id, pw, pwcert):
    os.system('taskkill /IM coStarter* /F /T')
    os.system('taskkill /IM CpStart* /F /T')
    os.system('taskkill /IM DibServer* /F /T')
    os.system('wmic process where "name like \'%coStarter%\'" call terminate')
    os.system('wmic process where "name like \'%CpStart%\'" call terminate')
    os.system('wmic process where "name like \'%DibServer%\'" call terminate')
    time.sleep(5)

    app = application.Application()
    app.start(f'C:\CREON\STARTER\coStarter.exe /prj:cp /id:{id} /pwd:{pw} /pwdcert:{pwcert} /autostart')
    time.sleep(10)
