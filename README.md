# FlightChecker

Windows GUI-App: Roundtrip-Vergleich (Ryanair cheapestPerDay)

Diese App sucht die günstigsten Ryanair-Flüge innerhalb einer Zeitspanne zum fliegen von CGN nach PMO bzw. von NRN nach TPS. Anschließend werden diese nochmal mithilfe eines Diagramms veranschaulicht gemacht.

Damit das Tool Flugpreise abrufen kann, benötigt es eine API-URL von Ryanair.
Diese ist absichtlich nicht im Code enthalten, da Ryanair bestimmte Nutzungsbedingungen hat und ich die URL aus rechtlichen Gründen nicht direkt mitverteilen möchte.

Bitte erstelle selbst die Datei:

data/ryanair_api.txt

und füge dort die Ryanair API-URL ein.

Die Verantwortung für die Nutzung der API liegt bei dir.
Bitte beachte die jeweiligen Bedingungen von Ryanair.

anschließend 

pyinstaller --noconfirm --clean --windowed --name RoundtripFinder --icon .\icon.ico --version-file .\version_info.txt --add-data "data\api.txt;data" .\src\main.py 

eingeben.